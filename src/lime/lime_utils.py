import os
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from tqdm import tqdm

load_dotenv()  # Load environment variables from .env file

data_dir = Path(os.getenv("DATA_DIR"))  # type: ignore


def get_norm_fn(norm_name):
    if norm_name == "euclidean_interpretable":

        def norm_fn(
            interpretable_inputs,
            segmentation_mask,
            path_to_image: str,
            small_size: bool = True,
        ) -> np.ndarray:
            interpretable_inputs = torch.tensor(interpretable_inputs)
            ones_vector = torch.ones_like(interpretable_inputs)
            norm = (
                torch.sum((interpretable_inputs - ones_vector) ** 2, dim=1)
                / interpretable_inputs.shape[1]
            )
            return norm.numpy()

    elif norm_name == "cosine_interpretable":

        def norm_fn(
            interpretable_inputs,
            segmentation_mask,
            path_to_image: str,
            small_size: bool = True,
        ) -> np.ndarray:
            interpretable_inputs = torch.tensor(interpretable_inputs)
            ones_vector = torch.ones_like(interpretable_inputs)
            norm = torch.nn.functional.cosine_similarity(
                interpretable_inputs, ones_vector, dim=1
            )
            return norm.numpy()

    elif norm_name == "euclidean_image":

        def norm_fn(
            interpretable_inputs,
            segmentation_mask,
            path_to_image: str,
            small_size: bool = True,
        ) -> np.ndarray:
            # Load image and preprocess it
            pil_image = Image.open(path_to_image).convert("RGB")
            image_tensor = F.to_tensor(pil_image)
            if small_size:
                image_tensor = F.resize(image_tensor, (224, 224))

                # Ensure interpolation is nearest for masks to preserve class integers
                segmentation_mask = (
                    F.resize(
                        segmentation_mask.unsqueeze(0).float(),
                        (224, 224),
                        interpolation=Image.NEAREST,
                    )
                    .squeeze(0)
                    .long()
                )

            # generate perturbed images
            perturbed_images = generate_perturbed_images(
                image_tensor, interpretable_inputs, segmentation_mask - 1
            )

            # compute norm
            num_samples = perturbed_images.shape[0]
            # Repeat original image to match num_samples
            original_image_expanded = image_tensor.repeat(num_samples, 1, 1, 1)

            # Compute Euclidean distances (L2 Norm)
            # Sum over C, H, W (dim 1, 2, 3)
            # compute by batch
            for i in tqdm(range(0, num_samples, 100)):
                batch_perturbed = perturbed_images[i : i + 100]
                batch_original = original_image_expanded[i : i + 100]
                batch_norm = torch.sqrt(
                    torch.sum((batch_original - batch_perturbed) ** 2, dim=(1, 2, 3))
                )
                if i == 0:
                    norm = batch_norm
                else:
                    norm = torch.cat((norm, batch_norm), dim=0)
            return norm.numpy()

    else:
        raise ValueError(
            f"Norm '{norm_name}' not recognized. Choose from 'euclidean_interpretable' or 'cosine_interpretable'."
        )
    return norm_fn


def get_from_interp_rep_transform(pixel_color: torch.Tensor):
    def from_interp_rep_transform(curr_sample, original_input, feature_mask, **kwargs):
        """
        Transform original input from interpretable representation by replacing the superpixel value by pixel_color where interpretable input is 0.
        Args:
            curr_sample (torch.Tensor): interpretable representation of shape (1, num_superpixels)
            original_input (torch.Tensor): original input of shape (1, 3, H, W)
            feature_mask (torch.Tensor): feature mask of shape (1, H, W)
        Returns:
            transformed_image (torch.Tensor): transformed image of shape (1, 3, H, W)
        """
        # search where curr_sample is 0
        superpixel_mask = np.where(curr_sample.cpu().numpy() == 0)[-1]
        # replace each 0 by pixel_color
        transformed_image = original_input.cpu().clone()
        for indice in superpixel_mask:
            selection = feature_mask == indice  # iterate over superpixels
            selection = selection.squeeze(0)  # shape (H, W)
            transformed_image[0, 0, selection] = pixel_color[0].item()
            transformed_image[0, 1, selection] = pixel_color[1].item()
            transformed_image[0, 2, selection] = pixel_color[2].item()
        return (
            transformed_image.cuda() if torch.cuda.is_available() else transformed_image
        )

    return from_interp_rep_transform


def get_pixel_color_for_perturbation(
    which_background_color: str,
    mean_normalization,
    std_normalization,
    image_path: Optional[Path] = None,
):
    if which_background_color == "black":
        pixel_color = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
    elif which_background_color == "white":
        pixel_color = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    elif which_background_color == "background":
        # load the image
        if image_path is None:
            raise ValueError("image_path must be provided when which_background_color is 'background'.")
        image = Image.open(image_path).convert("RGB")
        # highest number of pixels in the image
        data = Counter(
            [
                (c1, c2, c3)
                for c1, c2, c3 in zip(
                    np.ravel(np.array(image)[:, :, 0]),
                    np.ravel(np.array(image)[:, :, 1]),
                    np.ravel(np.array(image)[:, :, 2]),
                )
            ]
        )
        most_common_color = data.most_common(1)[0][0]
        pixel_color = (
            torch.tensor(most_common_color).view(3, 1, 1) / 255.0
        )  # normalize to [0, 1]
    else:
        raise ValueError(
            f"Background color '{which_background_color}' not recognized. Choose from 'black', 'white', or 'background'."
        )

    return v2.Normalize(mean=mean_normalization, std=std_normalization)(
        pixel_color
    )  # return normalized


def generate_perturbed_images(
    image, interpretable_inputs, segmented_image, pixel_color="black"
):
    """
    Args:
        image: (1, C, H, W) or (C, H, W)
        interpretable_inputs: (num_samples, num_superpixels)
        segmented_image: (H, W) containing superpixel IDs
    """
    num_samples = interpretable_inputs.shape[0]
    H, W = segmented_image.shape[-2:]

    # 1. Flatten the segmentation map to use it as a lookup index
    # Shape: (H * W)
    flat_indices = segmented_image.flatten()

    # 2. 'Gather' the on/off state for every pixel at once
    # We use the superpixel IDs in flat_indices to select columns from interpretable_inputs.
    # Logic: If input is not 0, we keep it (True). If 0, we blank it (False).
    # Shape: (num_samples, H * W)
    pixel_mask = interpretable_inputs[:, flat_indices] != 0

    # 3. Reshape and add channel dimension for broadcasting
    # Shape: (num_samples, 1, H, W)
    pixel_mask = pixel_mask.view(num_samples, 1, H, W)

    # 4. Apply the mask using element-wise multiplication
    # This automatically broadcasts 'image' across 'num_samples'
    # Shape: (num_samples, C, H, W)
    perturbed_images = image * pixel_mask

    return perturbed_images


def compute_similarities(original_image, perturbed_images, kernel_width: float):
    """
    Args:
        original_image: (1, C, H, W) or (C, H, W)
        perturbed_images: (num_samples, C, H, W)
    Returns:
        similarities: (num_samples, )
    """
    num_samples = perturbed_images.shape[0]
    # Repeat original image to match num_samples
    original_image_expanded = original_image.repeat(num_samples, 1, 1, 1)
    # Compute Euclidean distances
    similarities = torch.sqrt(
        torch.sum((original_image_expanded - perturbed_images) ** 2, dim=(1, 2, 3))
    )

    return torch.exp(-1 * (similarities**2) / (kernel_width**2))


def get_similarities(image, segmented_image, interpretable_inputs, kernel_width):
    perturbed_images = generate_perturbed_images(
        image, interpretable_inputs, segmented_image
    )
    similarities = compute_similarities(
        image, perturbed_images, kernel_width=kernel_width
    )
    return similarities


def load_data_for_training(
    dataset: str,
    model_to_be_explained: str,
    segmentation_algorithm: str,
    color: str,
    hash: str,
    which_walk: str,
    which_output: str,
):
    """
    This function loads the interpretable inputs and the outputs (logits or probas)
    for a given hash and according to random walk. It is used for training the
    interpretable model.
    """
    path_to_data = (
        Path(data_dir)
        / "inferences"
        / dataset
        / model_to_be_explained
        / segmentation_algorithm
        / color
        / which_walk
        / hash
    )

    if which_output == "logits":
        outputs = torch.load(
            path_to_data / "logits.pt", map_location=torch.device("cpu")
        )

    elif which_output == "probas":
        outputs = torch.load(
            path_to_data / "probas.pt", map_location=torch.device("cpu")
        )

    else:
        raise ValueError(
            f"Output type '{which_output}' not recognized. Choose from 'logits' or 'probas'."
        )

    interpretable_inputs = torch.load(
        path_to_data / "interpretable_inputs.pt", map_location=torch.device("cpu")
    )

    return interpretable_inputs, outputs


def perturb_specific_segments(
    image: torch.Tensor,
    segmented_image: torch.Tensor,
    indices_to_perturb: list[int],
    color: str = "black",
    background_not_used: bool = False,
    transform: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Perturbe des segments spécifiques d'une image.
    
    Args:
        image: Tensor de forme (C, H, W).
        segmented_image: Tensor de forme (H, W) contenant les IDs des superpixels.
        indices_to_perturb: Liste des IDs de superpixels à perturber (masquer).
        color: Couleur de perturbation ('median', 'mean', 'black', 'white').
        background_not_used: Si True, ajoute un 1 à la fin du vecteur de masque pour ignorer le fond.
        
    Returns:
        L'image perturbée sous forme de Tensor.
    """
    # 1. S'assurer que l'image est float [0, 1]
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
        
    segmented_image = segmented_image.long()
    num_segments: int = int(segmented_image.max().item() + 1)
    
    # 2. Définir la couleur de perturbation
    if color == "median":
        _median_color_x = image.median(dim=1)[0]
        pixel_color = torch.median(_median_color_x, dim=1)[0].view(3, 1, 1)
    elif color == "mean":
        _mean_color_x = image.mean(dim=(1, 2), keepdim=True)
        pixel_color = _mean_color_x
    elif color == "black":
        pixel_color = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
    elif color == "white":
        pixel_color = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    else:
        raise ValueError(
            f"Perturbation color '{color}' not recognized. Choose from 'median', 'mean', 'black', or 'white'."
        )
        
    pixel_color = pixel_color.to(image.device)
    
    # 3. Créer le vecteur de masque (1 pour garder, 0 pour perturber)
    mask_vector = torch.ones((num_segments,), dtype=torch.float32, device=image.device)
    for idx in indices_to_perturb:
        if idx < num_segments:
            mask_vector[idx] = 0.0
            
    if background_not_used:
        # Ajouter 1 à la fin du mask_vector pour éviter l'indexation négative
        mask_vector = torch.cat(
            (mask_vector, torch.tensor([1.0], device=image.device))
        )
        
    # 4. Créer le masque spatial
    # Projeter mask_vector en 2D (H, W), puis ajouter la dimension (1, H, W) pour le broadcasting
    spatial_mask = mask_vector[segmented_image].unsqueeze(0)
    
    # 5. Appliquer la perturbation
    perturbed_image = image * spatial_mask + pixel_color * (1 - spatial_mask)

    # 6. Appliquer la transformation de prétraitement si nécessaire
    if transform:
        perturbed_image = transform(perturbed_image)
    
    return perturbed_image


class PerturbationDataset(Dataset):
    def __init__(
        self,
        image: torch.Tensor,
        segmented_image: torch.Tensor,
        num_samples: int,
        color: str,
        random_walk: str = "uniform",
        transform: Optional[Callable] = None,
        background_not_used: bool = False,
    ) -> None:
        # 1. On s'assure que l'image est float [0, 1] dès le départ
        if image.dtype == torch.uint8:
            self.image = image.float() / 255.0
        else:
            self.image = image  # On suppose que c'est déjà float [0, 1]

        self.segmented_image = (
            segmented_image.long()
        )  # Doit être un entier (Long) pour l'indexation

        self.num_samples = num_samples
        self.num_segments: int = int(self.segmented_image.max().item() + 1)
        self.transform = transform
        self.random_walk = random_walk
        self.color = color
        self.background_not_used = background_not_used
        # define pixel color for perturbation
        if color == "median":
            _median_color_x = self.image.median(dim=1)[0]
            self.pixel_color = torch.median(_median_color_x, dim=1)[0].view(3, 1, 1)

        elif color == "mean":
            _mean_color_x = self.image.mean(dim=(1, 2), keepdim=True)
            self.pixel_color = _mean_color_x

        elif color == "black":
            self.pixel_color = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)

        elif color == "white":
            self.pixel_color = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)

        else:
            raise ValueError(
                f"Perturbation color '{color}' not recognized. Choose from 'median', 'mean', 'black', or 'white'."
            )

        print(f"Pixel color for perturbation: {self.pixel_color.squeeze().tolist()}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # 1. Générer le vecteur binaire de masquage (taille: num_segments)
        if idx == 0:
            # L'image originale (tous les segments sont actifs)
            mask_vector = torch.ones(
                (self.num_segments,), 
                dtype=torch.float32, 
                device=self.image.device
            )
        else:
            if self.random_walk == "uniform":
                # Masque aléatoire (Bernoulli distribution is faster/cleaner)
                mask_vector = torch.randint(
                    0,
                    2,
                    (self.num_segments,),
                    dtype=torch.float32,
                    device=self.image.device,
                )
            else:
                # TODO: implement other random walk strategies (Binomiale.)
                raise ValueError(
                    f"Random walk strategy '{self.random_walk}' not recognized. Choose from 'uniform'."
                )
        if self.background_not_used:
            # add 1 at the end of mask_vector to avoid negative indexing
            mask_vector = torch.cat(
                (mask_vector, torch.tensor([1.0], device=self.image.device))
            )

        # 2. Créer le masque spatial (Vectorisation)
        # On utilise segmented_image comme index pour projeter mask_vector en 2D (H, W)
        # mask_vector[segmented_image] va remplacer chaque pixel (valeur de segment)
        # par la valeur correspondante dans mask_vector (0 ou 1).
        spatial_mask = mask_vector[self.segmented_image]

        # 3. Broadcasting pour s'adapter aux canaux de l'image (C, H, W)
        # spatial_mask passe de (H, W) à (1, H, W) pour être multiplié avec (C, H, W)
        spatial_mask = spatial_mask.unsqueeze(0)

        # 4. Appliquer la perturbation
        # Les zones à 1 gardent l'image d'origine, les zones à 0 prennent la couleur de perturbation
        pixel_color_device = self.pixel_color.to(self.image.device)
        perturbed_image = self.image * spatial_mask + pixel_color_device * (
            1 - spatial_mask
        )

        # Note: Tu avais un argument 'transform' dans __init__ mais tu ne l'utilisais pas.
        # S'il y a une transformation à faire (ex: normalisation), c'est généralement ici :
        if self.transform:
            perturbed_image = self.transform(perturbed_image)

        return mask_vector, perturbed_image


def inference_on_perturbation_dataset(
    model: torch.nn.Module,
    perturbation_dataloader: DataLoader,
    background_not_used: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # inferences
    logits = []
    probas = []
    interpretable_inputs = []
    model.eval()
    if device is not None:
        model.to(device)
    with torch.no_grad():
        for batch in tqdm(perturbation_dataloader):
            mask_vectors, images = batch  # images shape: (B, C, H, W)
            if device is not None:
                images = images.to(device)
            outputs = model(images)  # outputs shape: (B, num_classes)
            logits.append(outputs.cpu())
            probas.append(torch.nn.functional.softmax(outputs, dim=1).cpu())
            interpretable_inputs.append(mask_vectors.cpu())
    logits = torch.cat(logits, dim=0)
    probas = torch.cat(probas, dim=0)
    interpretable_inputs = torch.cat(interpretable_inputs, dim=0)
    if background_not_used:
        interpretable_inputs = interpretable_inputs[:, :-1]  # remove background column
    return interpretable_inputs, logits, probas

def inference_on_image(
    model: torch.nn.Module,
    image: torch.Tensor
):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # check if image has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        probas = torch.nn.functional.softmax(model(image), dim=1)
    return probas.squeeze(0).cpu()
    

def explanation(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
):
    """
    Get LIME explanation for a specific class using weighted linear regression.

    :param dataset: Dataset for weighted linear regression
    :type dataset: TensorDataset with (interpretable_inputs #shape (num_examples, num_segments), outputs #shape (num_examples,), similarities #shape (num_examples,))
    :param batch_size: Size of batches for DataLoader
    :type batch_size: int

    :return: Coefficients of the weighted linear regression model
    :rtype: List[float]
    """
    if model == "linear":
        interpretable_model_instance = LinearRegression()
    else:
        raise ValueError(
            f"Model '{model}' not recognized. Choose from 'linear'."
        )

    # train the interpretable model with weights
    interpretable_model_instance.fit(X, y, sample_weight=weights)
    # compute the R^2 score
    r2_score = interpretable_model_instance.score(X, y, sample_weight=weights)

    # 3 - Return the coefficients as explanation
    return interpretable_model_instance.coef_.tolist(), r2_score
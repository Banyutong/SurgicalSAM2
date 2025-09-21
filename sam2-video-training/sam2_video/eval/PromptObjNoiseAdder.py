import numpy as np
import cv2
import random
import albumentations as A
from .utils import PromptObj


class PromptObjNoiseAdder:
    def __init__(self, bbox_noise_type="shift_scale", noise_intensity=0.1):
        self.BG_IMAGE = np.zeros((100, 100), dtype=np.uint8)
        self.bbox_noise_type = bbox_noise_type
        self.noise_intensity = noise_intensity
        self.MASK_TRANSFORM = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=self.noise_intensity,
                    scale_limit=self.noise_intensity,
                    rotate_limit=int(45 * self.noise_intensity),
                    p=0.5,
                ),
                A.OneOf(
                    [A.Lambda(mask=self.dilate_mask), A.Lambda(mask=self.erode_mask)],
                    p=0.5,
                ),
            ]
        )
        self.BBOX_TRANSFORM = self.get_bbox_transform()

    def get_bbox_transform(self):
        if self.bbox_noise_type == "shift":
            return A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=self.noise_intensity,
                        scale_limit=0,
                        rotate_limit=0,
                        p=0.5,
                    )
                ],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )
        elif self.bbox_noise_type == "scale":
            return A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=self.noise_intensity,
                        rotate_limit=0,
                        p=0.5,
                    )
                ],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )
        elif self.bbox_noise_type == "shift_scale":
            return A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=self.noise_intensity,
                        scale_limit=self.noise_intensity,
                        rotate_limit=0,
                        p=0.5,
                    )
                ],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )
        else:
            raise ValueError(
                "Invalid bbox_noise_type. Choose from 'shift', 'scale', or 'shift_scale'."
            )

    def dilate_mask(self, mask, **kwargs):
        kernel_size = random.randrange(3, 3 + int(21 * self.noise_intensity), 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        noised_mask = cv2.dilate(mask, kernel)
        return noised_mask.astype(bool)

    def erode_mask(self, mask, **kwargs):
        kernel_size = random.randrange(3, 3 + int(21 * self.noise_intensity), 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        noised_mask = cv2.erode(mask, kernel)
        return noised_mask.astype(bool)

    def add_noise_to_mask(self, obj: PromptObj):
        mask = obj.mask
        mask = mask.astype(np.uint8)
        transformed = self.MASK_TRANSFORM(image=self.BG_IMAGE, mask=mask)
        transformed_mask = transformed["mask"]
        obj.mask = transformed_mask.astype(bool)
        if obj.mask.sum() == 0:
            return None
        return obj

    def add_noise_to_bbox(self, obj: PromptObj):
        bbox = obj.bbox
        bbox.append("bbox")
        transformed = self.BBOX_TRANSFORM(image=self.BG_IMAGE, bboxes=[bbox])
        if len(transformed["bboxes"]) == 0:
            return None
        new_bbox = list(transformed["bboxes"][0])[:-1]
        obj.bbox = new_bbox
        return obj

    def add_noise_to_obj(self, obj: PromptObj, prompt_type: str):
        if self.BG_IMAGE.shape != obj.mask.shape:
            self.BG_IMAGE = np.zeros(obj.mask.shape, dtype=np.uint8)
        if prompt_type == "mask":
            return self.add_noise_to_mask(obj)
        elif prompt_type == "bbox":
            return self.add_noise_to_bbox(obj)

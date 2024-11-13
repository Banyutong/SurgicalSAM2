from PIL import Image


def crop(img, idxs):
    sub_imgs = []
    upper = 27
    lower = 282
    for idx in idxs:
        left = 10 + (idx - 1) * 446
        right = left + 446
        sub_imgs.append(img.crop((left, upper, right, lower)))

    return sub_imgs


def x_concat(sub_imgs):
    total_width = sum(sub_img.width for sub_img in sub_imgs)
    max_height = max(sub_img.height for sub_img in sub_imgs)

    # 创建一个新的图像，用于拼接子图像
    new_img = Image.new("RGB", (total_width, max_height))

    # 将子图像从左至右粘贴到新图像上
    x_offset = 0
    for sub_img in sub_imgs:
        new_img.paste(sub_img, (x_offset, 0))
        x_offset += sub_img.width

    return new_img


def y_concat(sub_imgs):
    total_height = sum(sub_img.height for sub_img in sub_imgs)
    max_width = max(sub_img.width for sub_img in sub_imgs)

    # 创建一个新的图像，用于拼接子图像
    new_img = Image.new("RGB", (max_width, total_height))

    # 将子图像从上至下粘贴到新图像上
    y_offset = 0
    for sub_img in sub_imgs:
        new_img.paste(sub_img, (0, y_offset))
        y_offset += sub_img.height

    return new_img

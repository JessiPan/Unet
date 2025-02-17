from PIL import Image
import os


def png_to_jpg(png_path, jpg_path):
    # 打开 PNG 图片
    img = Image.open(png_path)

    # 确保图片是 RGB 模式，因为 JPG 不支持透明度
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # 保存为 JPG 图片
    img.save(jpg_path, 'JPEG')
    print(f"Converted and saved: {jpg_path}")


if __name__ == "__main__":
    # 指定输入和输出文件夹
    input_folder = r"C:/Users/Jessi/Desktop/Dataset/Lovewd/Val/Urban/images_png"
    output_folder = r"C:/Users/Jessi/Desktop/Dataset/Lovewd/Val/Urban/JPEFGmages"

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹下的所有 PNG 文件
    for file in os.listdir(input_folder):
        if file.endswith(".png"):
            png_path = os.path.join(input_folder, file)
            jpg_path = os.path.join(output_folder, file[:-4] + ".jpg")
            png_to_jpg(png_path, jpg_path)
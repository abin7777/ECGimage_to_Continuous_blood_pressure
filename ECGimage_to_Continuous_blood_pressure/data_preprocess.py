import fitz  # PyMuPDF
from PIL import Image
import os
import shutil

# dir_path = 'zxb/'
dir_name = 'bbyz'
dir_path = dir_name + '/'
# 总的数据采集次数
total = 5

pdf_input_path = '/gjh/ECGtoBP/data_input/' + dir_path  # pdf目录
img_output_path = '/gjh/ECGtoBP/data_output/' + dir_path # 最终图片输出目录
temp_path = '/gjh/ECGtoBP/temp/'  # 临时目录


def extract_images_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print("输入目录不存在")
        return

    for filename in os.listdir(pdf_path):
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path + filename)
        name = filename.split('.')[0]

        # 确保输出目录存在

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        output_folder = temp_path

        # 遍历每一页
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)  # 获取页面上的所有图片

            # 对于每个图片
            for img_index, img in enumerate(image_list):
                xref = img[0]  # XREF编号
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # 图片扩展名

                # 保存图片
                image_filename = f"{page_num + 1}.{image_ext}"
                image_filepath = os.path.join(output_folder, image_filename)

                # 取第二个图片
                if img_index == 1:
                    with open(image_filepath, "wb") as image_file:
                        image_file.write(image_bytes)
                    print(f"获取到图片 {image_filepath}")


def split_image_horizontally(filename):
    data = []

    images = []

    # 打开图片
    # if 'total' in filename:

    img = Image.open(temp_path + filename)
    width, height = img.size

    # 确保输出目录存在
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # 计算每个部分的高度
    parts = 9
    part_height = 160

    # 分割图片
    for i in range(parts):
        # 计算当前部分的上边缘和下边缘
        upper = i * part_height
        lower = (i + 1) * part_height + 1

        # 裁剪图片
        if i == 8:
            print(width)
            part = img.crop((0, upper, 801, lower))
        else:
            part = img.crop((0, upper, width, lower))

        images.append(part)
        # 保存分割后的图片
        file_name = f"{filename.split('.')[0]}"
        # part_filename = f"{filename.split('.')[0].strip('total')}part_{i + 1}.png"
        # part_filepath = os.path.join(temp_path, part_filename)
        # part.save(part_filepath, 'PNG')
        # print(f"Saved {part_filepath}")
        data.append(file_name)
        data.append(images)
    return data


def concatenate_images_horizontally(data, output_path, i):
    # 确保至少有一张图片
    if not data[1]:
        print("没有图片可以拼接。")
        return

    # 计算总宽度和最大高度
    total_width = sum(img.width for img in data[1])
    max_height = max(img.height for img in data[1])

    # 创建一个新的空白图像，用于存放拼接后的结果
    result = Image.new('RGB', (total_width, max_height))

    # 拼接图片
    x_offset = 0
    for img in data[1]:
        # 如果图片的高度与最大高度不一致，则调整图片大小
        if img.height != max_height:
            img = img.resize((img.width, max_height), Image.LANCZOS)

        # 将当前图片粘贴到结果图像的适当位置
        result.paste(img, (x_offset, 0))
        x_offset += img.width

        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

    # 保存拼接后的图像
    result.save(output_path + data[0] + ".png", 'PNG')
    print(f"拼接后的图片已保存至 {output_path}")


def dealImages_from_pdf(pdf_path, output_folder):
    # 判别输入目录是否存在
    if not pdf_path:
        os.makedirs(pdf_path)
        print(f"输入目录不存在，已创建输入目录，请将pdf文件放在以下目录： {pdf_path}")

    i = 1
    # 遍历提取pdf内文件
    extract_images_from_pdf(pdf_path)
    for file_name in os.listdir(temp_path):
        concatenate_images_horizontally(split_image_horizontally(file_name), output_folder, i)
        i = i + 1

    # 删除临时目录
    shutil.rmtree(temp_path)


# 执行主函数
dealImages_from_pdf(pdf_input_path, img_output_path)



def crop_images(input_dir, output_dir, target_width):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整的文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 打开图像
            image = Image.open(input_path)

            # 获取图像的宽度和高度
            width, height = image.size

            # 定义裁剪区域
            left = 0
            top = 0
            right = target_width
            bottom = height

            # 裁剪图像
            cropped_image = image.crop((left, top, right, bottom))

            # 保存裁剪后的图像
            cropped_image.save(output_path)
            print(f"Cropped and saved: {output_path}")

# 设置输入和输出目录
input_directory = '/gjh/ECGtoBP/data_output/' + dir_name
output_directory = '/gjh/ECGtoBP/data_output_reshaped/' + dir_name
target_width = 12000

# 调用函数
crop_images(input_directory, output_directory, target_width)



import pandas as pd

def sliding_window_crop(image_path, output_dir, window_width=140, step_size=2, crop_height=80, excel_file_path=None, num=None):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开图片
    image = Image.open(image_path)
    width, height = image.size

    # 计算中间80个像素的高度范围
    start_y = (height - crop_height) // 2
    end_y = start_y + crop_height

    # 计算滑动次数
    num_steps = (width - window_width) // step_size + 1

    all_sheets = pd.read_excel(excel_file_path, sheet_name=None, usecols=['Pressure [mmHg]'])
    y_datas = all_sheets['Sheet' + num][70:]

    # 开始滑动窗口裁剪
    for i in range(num_steps):
        # 计算窗口的位置
        left = i * step_size
        top = start_y
        right = left + window_width
        bottom = end_y

        # 裁剪窗口内的图片
        cropped_image = image.crop((left, top, right, bottom))

        y = y_datas.iloc[i]
        y = float(y)
        # 保存裁剪后的图片
        output_path = f"{output_dir}/{num}_{i:04d}_{y}.png"
        cropped_image.save(output_path)
        print(f"Saved {output_path}")

# 示例用法
for i in range(1, total+1):
    num = str(i)
    image_path = "/gjh/ECGtoBP/data_output_reshaped/" + dir_path + num +".png"
    output_dir = '/gjh/ECGtoBP/ECG_images_with_bp/' + dir_name
    excel_file_path = '/gjh/ECGtoBP/BPdatas/' + dir_path + 'BPdata.xlsx'
    sliding_window_crop(image_path, output_dir, window_width=140, step_size=2, crop_height=80, excel_file_path=excel_file_path, num=num)


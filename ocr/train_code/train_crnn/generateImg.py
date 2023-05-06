from PIL import Image, ImageDraw, ImageFont
import random

# 可以自行替换为其他字体文件
font_path = '/path/to/font.ttf'
font_size = 32  # 字体大小
image_width = 256  # 图片宽度
image_height = 64  # 图片高度
num_samples = 10000  # 样本数量
data_file = 'data.txt'  # 数据保存文件名

characters = []
# 生成所有可能的中文字符
for i in range(0x4E00, 0x9FBF + 1):
    characters.append(chr(i))

with open(data_file, 'w', encoding='utf-8') as f:
    for i in range(num_samples):
        c = random.choice(characters)  # 随机选择一个中文字符
        img = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(c, font=font)
        x = (image_width - text_width) / 2
        y = (image_height - text_height) / 2
        draw.text((x, y), c, font=font, fill=(0, 0, 0))  # 将文字绘制到图片上
        image_file = 'image_{}.jpg'.format(i)
        img.save(image_file)  # 保存图片
        f.write('{}\t{}\n'.format(image_file, c))  # 将图片路径和标签写入数据文件中
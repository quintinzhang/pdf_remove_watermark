from flask import Flask, render_template, request, send_file, jsonify, session
import threading
import time
import os
import cv2
import numpy as np
import fitz
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
from fpdf import FPDF
from PIL import Image
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import re
DEFAULT_DPI = 300

app = Flask(__name__)
app.secret_key = 'watermark_removal_progress_key'

# 全局进度存储
progress_data = {}


# 图像去除水印函数
def remove_watermark(image_path):
    img = cv2.imread(image_path)
    lower_hsv = np.array([160, 160, 160])
    upper_hsv = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_hsv, upper_hsv)
    mask = cv2.GaussianBlur(mask, (1, 1), 0)
    img[mask == 255] = [255, 255, 255]
    cv2.imwrite(image_path, img)


# 将PDF转换为图片，并保存到指定目录

def pdf_to_images(pdf_path, output_folder, dpi=DEFAULT_DPI, progress_id=None):
    images = []
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    
    # 记录开始时间
    start_time = time.time()
    
    for page_num in range(total_pages):
        page_start_time = time.time()
        
        page = doc[page_num]
        # 使用传入的DPI设置
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        images.append(image_path)
        # 去除每张图片的水印
        remove_watermark(image_path)
        
        # 更新进度和时间估算
        if progress_id:
            progress = int((page_num + 1) / total_pages * 75)  # PDF转换占75%
            
            # 计算剩余时间
            elapsed_time = time.time() - start_time
            if page_num > 0:
                avg_time_per_page = elapsed_time / (page_num + 1)
                remaining_pages = total_pages - (page_num + 1)
                estimated_remaining = int(avg_time_per_page * remaining_pages + 10)  # +10秒用于PDF生成
            else:
                estimated_remaining = int(total_pages * 3)  # 初始估算每页3秒
            
            progress_data[progress_id] = {
                'progress': progress, 
                'status': f'处理第{page_num + 1}/{total_pages}页',
                'estimated_remaining': estimated_remaining
            }
            
    return images


# 将图片合并为PDF

# 定义A4纸张在72dpi下的像素尺寸（宽度和高度）
A4_SIZE_PX_72DPI = (595, 842)


def images_to_pdf(image_paths, output_path):
    pdf_writer = FPDF(unit='pt', format='A4')

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size

            # 计算实际DPI（假设从pdf转图片时已设置为300 DPI）
            dpi = 300
            ratio = min(A4_SIZE_PX_72DPI[0] / width, A4_SIZE_PX_72DPI[1] / height)

            # 缩放图像以适应A4纸张，并保持长宽比
            img_resized = img.resize((int(width * ratio), int(height * ratio)))

            # 创建临时文件并写入图片数据
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                img_resized.save(temp_file.name, format='PNG')

            # 添加一页
            pdf_writer.add_page()

            # 使用临时文件路径添加图像到PDF
            pdf_writer.image(temp_file.name, x=0, y=0, w=A4_SIZE_PX_72DPI[0], h=A4_SIZE_PX_72DPI[1])

    # 清理临时文件
    for image_path in image_paths:
        _, temp_filename = os.path.split(image_path)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    pdf_writer.output(output_path)


# 直接PDF去水印功能 - 基于文本和对象检测
def direct_pdf_watermark_removal(input_path, output_path, watermark_keywords=None, progress_id=None):
    """
    直接在PDF文档层面去除水印，不通过图片转换
    支持去除文本水印、透明对象、特定位置的内容等
    """
    if watermark_keywords is None:
        watermark_keywords = ['水印', 'watermark', 'CONFIDENTIAL', '机密', '内部资料', 'DRAFT', '草稿']
    
    try:
        doc = fitz.open(input_path)
        total_pages = doc.page_count
        removed_count = 0
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # 更新进度
            if progress_id:
                progress = int((page_num / total_pages) * 90)  # 90%用于页面处理
                progress_data[progress_id] = {
                    'progress': progress,
                    'status': f'处理第{page_num + 1}/{total_pages}页 (直接模式)',
                    'estimated_remaining': int((total_pages - page_num) * 0.5)
                }
            
            # 方法1：移除包含水印关键词的文本
            text_instances = page.search_for('')  # 获取所有文本
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # 检查是否包含水印关键词
                            for keyword in watermark_keywords:
                                if keyword.lower() in text.lower():
                                    # 创建覆盖矩形（白色背景）
                                    bbox = span["bbox"]
                                    rect = fitz.Rect(bbox)
                                    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                                    removed_count += 1
            
            # 方法2：移除透明度很高的对象（通常是水印）
            annotations = page.annots()
            for annot in annotations:
                # 移除透明注释和水印注释
                if annot.type[1] in ['Watermark', 'Stamp']:
                    page.delete_annot(annot)
                    removed_count += 1
            
            # 方法3：移除特定位置的重复内容（如页眉页脚水印）
            # 检测页面边缘区域的重复文本
            page_rect = page.rect
            margin = 50  # 边缘50点范围内
            
            # 页面四角和中心区域水印检测
            watermark_regions = [
                fitz.Rect(0, 0, page_rect.width/3, margin),  # 顶部左
                fitz.Rect(page_rect.width*2/3, 0, page_rect.width, margin),  # 顶部右
                fitz.Rect(0, page_rect.height-margin, page_rect.width/3, page_rect.height),  # 底部左
                fitz.Rect(page_rect.width*2/3, page_rect.height-margin, page_rect.width, page_rect.height),  # 底部右
                fitz.Rect(page_rect.width/3, page_rect.height/3, page_rect.width*2/3, page_rect.height*2/3)  # 中心
            ]
            
            for region in watermark_regions:
                region_text = page.get_text("text", clip=region).strip()
                if region_text and len(region_text) < 50:  # 短文本可能是水印
                    for keyword in watermark_keywords:
                        if keyword.lower() in region_text.lower():
                            page.draw_rect(region, color=(1, 1, 1), fill=(1, 1, 1))
                            removed_count += 1
                            break
        
        # 保存处理后的PDF
        doc.save(output_path)
        doc.close()
        
        # 最终进度更新
        if progress_id:
            progress_data[progress_id] = {
                'progress': 100,
                'status': f'处理完成！共移除 {removed_count} 个水印对象',
                'estimated_remaining': 0
            }
        
        return True, removed_count
        
    except Exception as e:
        if progress_id:
            progress_data[progress_id] = {
                'progress': 0,
                'status': f'处理失败: {str(e)}',
                'estimated_remaining': 0
            }
        return False, 0


def process_direct_watermark_removal(progress_id, watermark_text=""):
    """后台处理直接PDF水印去除任务"""
    try:
        pdf_path = 'uploads/uploaded_file.pdf'
        output_pdf_path = 'output_file_direct.pdf'
        
        # 解析用户提供的水印关键词
        custom_keywords = []
        if watermark_text.strip():
            custom_keywords = [kw.strip() for kw in watermark_text.split(',') if kw.strip()]
        
        # 使用默认关键词 + 用户自定义关键词
        default_keywords = ['水印', 'watermark', 'CONFIDENTIAL', '机密', '内部资料', 'DRAFT', '草稿']
        all_keywords = default_keywords + custom_keywords
        
        # 初始进度
        progress_data[progress_id] = {
            'progress': 5,
            'status': '开始分析PDF文档...',
            'estimated_remaining': 10
        }
        
        success, removed_count = direct_pdf_watermark_removal(
            pdf_path, 
            output_pdf_path, 
            all_keywords, 
            progress_id
        )
        
        if success:
            progress_data[progress_id] = {
                'progress': 100,
                'status': f'直接处理完成！共移除 {removed_count} 个水印',
                'estimated_remaining': 0
            }
        
    except Exception as e:
        progress_data[progress_id] = {
            'progress': 0,
            'status': f'处理失败: {str(e)}',
            'estimated_remaining': 0
        }



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        pdf_path = 'uploads/uploaded_file.pdf'
        uploaded_file.save(pdf_path)
        return render_template('index.html', message='文件上传成功')


def process_watermark_removal(progress_id, dpi=DEFAULT_DPI):
    """后台处理水印去除任务"""
    try:
        pdf_path = 'uploads/uploaded_file.pdf'
        output_folder = 'output_images'
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取总页数进行初始时间估算
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        estimated_total_time = total_pages * 3 + 10  # 每页3秒 + 10秒生成
        
        # 更新进度：开始转换PDF
        progress_data[progress_id] = {
            'progress': 5, 
            'status': '开始转换PDF...',
            'estimated_remaining': estimated_total_time
        }
        
        image_paths = pdf_to_images(pdf_path, output_folder, dpi, progress_id)
        
        # 更新进度：开始合并PDF
        progress_data[progress_id] = {
            'progress': 85, 
            'status': '生成PDF文件...',
            'estimated_remaining': 5
        }
        
        output_pdf_path = 'output_file.pdf'
        images_to_pdf(image_paths, output_pdf_path)
        
        # 完成
        progress_data[progress_id] = {
            'progress': 100, 
            'status': '处理完成！',
            'estimated_remaining': 0
        }
        
    except Exception as e:
        progress_data[progress_id] = {
            'progress': 0, 
            'status': f'处理失败: {str(e)}',
            'estimated_remaining': 0
        }


@app.route('/remove_watermark', methods=['POST'])
def remove_watermark_route():
    import uuid
    progress_id = str(uuid.uuid4())
    session['progress_id'] = progress_id
    
    # 获取DPI参数，默认为300
    dpi = int(request.form.get('dpi', DEFAULT_DPI))
    # 确保DPI在合理范围内
    dpi = max(72, min(600, dpi))
    
    # 初始化进度
    progress_data[progress_id] = {'progress': 0, 'status': '开始处理...'}
    
    # 启动后台线程处理
    thread = threading.Thread(target=process_watermark_removal, args=(progress_id, dpi))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': '开始处理', 'dpi': dpi})


@app.route('/remove_watermark_direct', methods=['POST'])
def remove_watermark_direct_route():
    import uuid
    progress_id = str(uuid.uuid4())
    session['progress_id'] = progress_id
    
    # 获取用户自定义的水印文本
    watermark_text = request.form.get('watermark_text', '')
    
    # 初始化进度
    progress_data[progress_id] = {'progress': 0, 'status': '开始直接处理...'}
    
    # 启动后台线程处理
    thread = threading.Thread(target=process_direct_watermark_removal, args=(progress_id, watermark_text))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': '开始直接处理', 'method': 'direct'})


@app.route('/download_direct')
def download_direct():
    output_pdf_path = 'output_file_direct.pdf'
    return send_file(output_pdf_path, as_attachment=True, download_name='去水印PDF(直接模式).pdf')


@app.route('/progress')
def get_progress():
    progress_id = session.get('progress_id')
    if progress_id and progress_id in progress_data:
        return jsonify(progress_data[progress_id])
    return jsonify({'progress': 0, 'status': '未开始'})


@app.route('/download')
def download():
    output_pdf_path = 'output_file.pdf'
    return send_file(output_pdf_path, as_attachment=True)


@app.route('/UI1.html')
def ui1():
    return send_file(os.path.abspath('UI1.html'))


@app.route('/UI.html')
def ui():
    return send_file(os.path.abspath('UI.html'))


@app.route('/try_pdf.html')
def try_pdf():
    return send_file(os.path.abspath('try_pdf.html'))


@app.route('/try_pdf_fetch.html')
def try_pdf_fetch():
    return send_file(os.path.abspath('try_pdf_fetch.html'))


if __name__ == '__main__':
    app.run(debug=True)

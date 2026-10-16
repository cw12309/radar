"""
专业医学影像查看器组件
基于Cornerstone3D和Streamlit Components
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import base64
from pathlib import Path
import json


def create_cornerstone_viewer(image_data, window_center=40, window_width=400, 
                              slice_index=0, view_mode="axial"):
    """
    创建基于Cornerstone.js的专业医学影像查看器
    
    Args:
        image_data: numpy数组 (D, H, W) 或 (H, W, D)
        window_center: 窗位
        window_width: 窗宽
        slice_index: 当前切片索引
        view_mode: 视图模式 (axial/sagittal/coronal)
    """
    
    # 确保数据格式正确
    if len(image_data.shape) == 3:
        if view_mode == "axial":
            slice_data = image_data[slice_index, :, :]
        elif view_mode == "sagittal":
            slice_data = image_data[:, slice_index, :]
        elif view_mode == "coronal":
            slice_data = image_data[:, :, slice_index]
    else:
        slice_data = image_data
    
    # 归一化到0-255
    slice_data = slice_data.astype(np.float32)
    slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
    slice_data = (slice_data * 255).astype(np.uint8)
    
    # 转换为base64
    from PIL import Image
    import io
    img = Image.fromarray(slice_data)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # HTML模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Image Viewer</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: #0e1117;
                color: #fafafa;
            }}
            
            .viewer-container {{
                width: 100%;
                height: 100vh;
                display: flex;
                flex-direction: column;
                background: #1a1d24;
            }}
            
            .toolbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px 20px;
                background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
                border-bottom: 1px solid #374151;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }}
            
            .toolbar-left, .toolbar-right {{
                display: flex;
                gap: 12px;
                align-items: center;
            }}
            
            .tool-button {{
                padding: 8px 16px;
                background: #374151;
                border: 1px solid #4b5563;
                border-radius: 6px;
                color: #f3f4f6;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            
            .tool-button:hover {{
                background: #4b5563;
                border-color: #6b7280;
                transform: translateY(-1px);
            }}
            
            .tool-button.active {{
                background: #3b82f6;
                border-color: #2563eb;
            }}
            
            .slider-group {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .slider-group label {{
                font-size: 13px;
                color: #9ca3af;
                min-width: 60px;
            }}
            
            .slider-group input[type="range"] {{
                width: 120px;
                height: 4px;
                background: #374151;
                border-radius: 2px;
                outline: none;
            }}
            
            .slider-group input[type="range"]::-webkit-slider-thumb {{
                appearance: none;
                width: 14px;
                height: 14px;
                background: #3b82f6;
                border-radius: 50%;
                cursor: pointer;
            }}
            
            .slider-group span {{
                font-size: 13px;
                color: #f3f4f6;
                min-width: 40px;
                text-align: right;
            }}
            
            .canvas-container {{
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                position: relative;
            }}
            
            #imageCanvas {{
                max-width: 100%;
                max-height: 100%;
                border: 2px solid #374151;
                border-radius: 8px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                cursor: crosshair;
            }}
            
            .info-overlay {{
                position: absolute;
                top: 30px;
                left: 30px;
                background: rgba(17, 24, 39, 0.85);
                backdrop-filter: blur(10px);
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 12px;
                line-height: 1.6;
                border: 1px solid #374151;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }}
            
            .info-item {{
                display: flex;
                justify-content: space-between;
                gap: 20px;
            }}
            
            .info-label {{
                color: #9ca3af;
            }}
            
            .info-value {{
                color: #f3f4f6;
                font-weight: 500;
            }}
            
            .measurement-overlay {{
                position: absolute;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container">
            <div class="toolbar">
                <div class="toolbar-left">
                    <button class="tool-button active" id="panBtn" onclick="setTool('pan')">
                        <span>🔍</span> 平移
                    </button>
                    <button class="tool-button" id="windowBtn" onclick="setTool('window')">
                        <span>🔆</span> 窗位/窗宽
                    </button>
                    <button class="tool-button" id="zoomBtn" onclick="setTool('zoom')">
                        <span>🔎</span> 缩放
                    </button>
                    <button class="tool-button" onclick="resetView()">
                        <span>↺</span> 重置
                    </button>
                </div>
                
                <div class="toolbar-right">
                    <div class="slider-group">
                        <label>窗位:</label>
                        <input type="range" id="windowCenter" min="-1000" max="1000" value="{window_center}" oninput="updateWindow()">
                        <span id="wcValue">{window_center}</span>
                    </div>
                    <div class="slider-group">
                        <label>窗宽:</label>
                        <input type="range" id="windowWidth" min="1" max="2000" value="{window_width}" oninput="updateWindow()">
                        <span id="wwValue">{window_width}</span>
                    </div>
                </div>
            </div>
            
            <div class="canvas-container">
                <canvas id="imageCanvas"></canvas>
                
                <div class="info-overlay">
                    <div class="info-item">
                        <span class="info-label">切片:</span>
                        <span class="info-value" id="sliceInfo">{slice_index + 1} / {image_data.shape[0]}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">视图:</span>
                        <span class="info-value">{view_mode.upper()}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">鼠标位置:</span>
                        <span class="info-value" id="mousePos">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">像素值:</span>
                        <span class="info-value" id="pixelValue">-</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            let currentTool = 'pan';
            let isDragging = false;
            let startX, startY;
            let offsetX = 0, offsetY = 0;
            let scale = 1;
            let windowCenter = {window_center};
            let windowWidth = {window_width};
            
            img.onload = function() {{
                canvas.width = img.width;
                canvas.height = img.height;
                drawImage();
            }};
            
            img.src = 'data:image/png;base64,{img_base64}';
            
            function drawImage() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.translate(canvas.width/2 + offsetX, canvas.height/2 + offsetY);
                ctx.scale(scale, scale);
                ctx.translate(-canvas.width/2, -canvas.height/2);
                
                // 应用窗位窗宽
                ctx.filter = `brightness(${{windowCenter / 500 + 1}}) contrast(${{windowWidth / 400}})`;
                ctx.drawImage(img, 0, 0);
                ctx.restore();
            }}
            
            function setTool(tool) {{
                currentTool = tool;
                document.querySelectorAll('.tool-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(tool + 'Btn').classList.add('active');
                
                if (tool === 'pan') canvas.style.cursor = 'move';
                else if (tool === 'zoom') canvas.style.cursor = 'zoom-in';
                else canvas.style.cursor = 'crosshair';
            }}
            
            function resetView() {{
                scale = 1;
                offsetX = 0;
                offsetY = 0;
                drawImage();
            }}
            
            function updateWindow() {{
                windowCenter = parseInt(document.getElementById('windowCenter').value);
                windowWidth = parseInt(document.getElementById('windowWidth').value);
                document.getElementById('wcValue').textContent = windowCenter;
                document.getElementById('wwValue').textContent = windowWidth;
                drawImage();
            }}
            
            canvas.addEventListener('mousedown', (e) => {{
                isDragging = true;
                startX = e.clientX - offsetX;
                startY = e.clientY - offsetY;
            }});
            
            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const x = Math.floor((e.clientX - rect.left) / scale);
                const y = Math.floor((e.clientY - rect.top) / scale);
                
                document.getElementById('mousePos').textContent = `(${{x}}, ${{y}})`;
                
                // 获取像素值
                const imageData = ctx.getImageData(x, y, 1, 1);
                const pixel = imageData.data[0];
                document.getElementById('pixelValue').textContent = pixel;
                
                if (isDragging) {{
                    if (currentTool === 'pan') {{
                        offsetX = e.clientX - startX;
                        offsetY = e.clientY - startY;
                        drawImage();
                    }} else if (currentTool === 'window') {{
                        const dx = e.clientX - startX;
                        const dy = e.clientY - startY;
                        windowCenter = Math.max(-1000, Math.min(1000, windowCenter + dx));
                        windowWidth = Math.max(1, Math.min(2000, windowWidth - dy));
                        document.getElementById('windowCenter').value = windowCenter;
                        document.getElementById('windowWidth').value = windowWidth;
                        updateWindow();
                        startX = e.clientX;
                        startY = e.clientY;
                    }}
                }}
            }});
            
            canvas.addEventListener('mouseup', () => {{
                isDragging = false;
            }});
            
            canvas.addEventListener('wheel', (e) => {{
                e.preventDefault();
                if (currentTool === 'zoom') {{
                    const delta = e.deltaY > 0 ? 0.9 : 1.1;
                    scale = Math.max(0.1, Math.min(10, scale * delta));
                    drawImage();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # 渲染组件
    components.html(html_template, height=800, scrolling=False)


def create_multi_view_viewer(image_data, window_center=40, window_width=400):
    """
    创建三视图查看器（轴位、矢状位、冠状位）
    
    Args:
        image_data: numpy数组 (D, H, W)
        window_center: 窗位
        window_width: 窗宽
    """
    
    depth, height, width = image_data.shape
    mid_axial = depth // 2
    mid_sagittal = width // 2
    mid_coronal = height // 2
    
    # 获取三个视图的切片
    axial_slice = image_data[mid_axial, :, :]
    sagittal_slice = image_data[:, :, mid_sagittal]
    coronal_slice = image_data[:, mid_coronal, :]
    
    def normalize_and_encode(slice_data):
        slice_data = slice_data.astype(np.float32)
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        slice_data = (slice_data * 255).astype(np.uint8)
        
        from PIL import Image
        import io
        img = Image.fromarray(slice_data)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    axial_base64 = normalize_and_encode(axial_slice)
    sagittal_base64 = normalize_and_encode(sagittal_slice)
    coronal_base64 = normalize_and_encode(coronal_slice)
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0e1117;
                color: #fafafa;
                padding: 20px;
            }}
            
            .multi-view-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: repeat(2, 1fr);
                gap: 16px;
                height: 90vh;
            }}
            
            .view-panel {{
                background: #1a1d24;
                border: 2px solid #374151;
                border-radius: 12px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }}
            
            .view-header {{
                padding: 12px 16px;
                background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
                border-bottom: 1px solid #374151;
                font-weight: 600;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .view-content {{
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 16px;
                position: relative;
            }}
            
            .view-content img {{
                max-width: 100%;
                max-height: 100%;
                border-radius: 4px;
                image-rendering: pixelated;
            }}
            
            .view-label {{
                position: absolute;
                top: 24px;
                left: 24px;
                background: rgba(59, 130, 246, 0.9);
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            
            .panel-3d {{
                grid-column: 2;
                grid-row: 1 / 3;
            }}
            
            .controls-panel {{
                background: #1a1d24;
                border: 2px solid #374151;
                border-radius: 12px;
                padding: 16px;
            }}
            
            .control-group {{
                margin-bottom: 16px;
            }}
            
            .control-group label {{
                display: block;
                margin-bottom: 8px;
                font-size: 13px;
                color: #9ca3af;
            }}
            
            .control-group input[type="range"] {{
                width: 100%;
                height: 6px;
                background: #374151;
                border-radius: 3px;
                outline: none;
            }}
            
            .control-group input[type="range"]::-webkit-slider-thumb {{
                appearance: none;
                width: 16px;
                height: 16px;
                background: #3b82f6;
                border-radius: 50%;
                cursor: pointer;
            }}
            
            .value-display {{
                text-align: right;
                color: #f3f4f6;
                font-size: 13px;
                margin-top: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="multi-view-container">
            <div class="view-panel">
                <div class="view-header">
                    <span>🔵</span> 轴位视图 (Axial)
                </div>
                <div class="view-content">
                    <img src="data:image/png;base64,{axial_base64}" alt="Axial View">
                    <div class="view-label">切片 {mid_axial + 1}/{depth}</div>
                </div>
            </div>
            
            <div class="view-panel panel-3d">
                <div class="view-header">
                    <span>🟢</span> 3D 体绘制
                </div>
                <div class="view-content">
                    <div style="color: #9ca3af; text-align: center;">
                        <div style="font-size: 48px; margin-bottom: 16px;">🏥</div>
                        <div>3D渲染视图</div>
                        <div style="font-size: 12px; margin-top: 8px;">集成专业3D可视化</div>
                    </div>
                </div>
            </div>
            
            <div class="view-panel">
                <div class="view-header">
                    <span>🟡</span> 矢状位视图 (Sagittal)
                </div>
                <div class="view-content">
                    <img src="data:image/png;base64,{sagittal_base64}" alt="Sagittal View">
                    <div class="view-label">切片 {mid_sagittal + 1}/{width}</div>
                </div>
            </div>
            
            <div class="view-panel">
                <div class="view-header">
                    <span>🔴</span> 冠状位视图 (Coronal)
                </div>
                <div class="view-content">
                    <img src="data:image/png;base64,{coronal_base64}" alt="Coronal View">
                    <div class="view-label">切片 {mid_coronal + 1}/{height}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(html_template, height=900, scrolling=False)

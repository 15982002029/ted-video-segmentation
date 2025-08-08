# 流程图使用说明

## 📋 文件说明
- `01_智能爬取阶段.mmd`: 智能爬取阶段的Mermaid代码
- `02_安全处理阶段.mmd`: 安全处理阶段的Mermaid代码

## 🌐 在线生成图片的方法

### 方法1: Mermaid Live Editor (推荐)
1. 访问: https://mermaid.live/
2. 复制 .mmd 文件中的代码到左侧编辑器
3. 右侧会实时显示流程图
4. 点击右上角的下载按钮保存为PNG/SVG

### 方法2: GitHub Mermaid
1. 在GitHub中创建一个Markdown文件
2. 使用以下格式包装代码:
```mermaid
[复制.mmd文件中的代码]
```
3. GitHub会自动渲染流程图

### 方法3: Mermaid CLI (需要Node.js)
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i 01_智能爬取阶段.mmd -o 01_智能爬取阶段.png
mmdc -i 02_安全处理阶段.mmd -o 02_安全处理阶段.svg
```

### 方法4: VS Code插件
1. 安装 "Mermaid Preview" 插件
2. 打开 .mmd 文件
3. 使用 Ctrl+Shift+P 搜索 "Mermaid Preview"
4. 可以预览和导出图片

## 🎨 自定义样式
您可以修改 .mmd 文件中的 style 行来改变颜色:
- `fill:#e1f5fe` 改变填充颜色
- `stroke:#000000` 改变边框颜色
- `color:#000000` 改变文字颜色

## 📱 建议的输出格式
- **PPT/文档**: 使用SVG格式 (矢量图，可无限缩放)
- **网页/分享**: 使用PNG格式 (位图，兼容性好)

"""
Simple HTML converter for MLOps project documentation
"""

import markdown2
import base64
import os
from datetime import datetime

def embed_images_in_markdown(markdown_content):
    """Convert image references to base64 embedded images."""
    import re
    
    # Find all image references in markdown
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_img(match):
        alt_text = match.group(1)
        img_src = match.group(2)
        
        # Check if image file exists
        if os.path.exists(img_src):
            try:
                # Read image and convert to base64
                with open(img_src, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Determine image type
                img_ext = os.path.splitext(img_src)[1].lower()
                if img_ext == '.png':
                    img_type = 'image/png'
                elif img_ext in ['.jpg', '.jpeg']:
                    img_type = 'image/jpeg'
                else:
                    img_type = 'image/png'  # default
                
                # Create embedded image HTML
                return f'<img src="data:{img_type};base64,{img_base64}" alt="{alt_text}" style="max-width:100%; height:auto; display:block; margin:20px auto; border:1px solid #ddd; border-radius:5px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">'
            except Exception as e:
                print(f"Error processing image {img_src}: {e}")
                return f'<img src="{img_src}" alt="{alt_text}" style="max-width:100%; height:auto;">'
        else:
            return f'<img src="{img_src}" alt="{alt_text}" style="max-width:100%; height:auto;">'
    
    return re.sub(img_pattern, replace_img, markdown_content)

def create_html_document():
    """Create the complete HTML documentation."""
    
    # Read the markdown file
    with open('project_doc.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Embed images
    markdown_content = embed_images_in_markdown(markdown_content)
    
    # Convert to HTML
    html_content = markdown2.markdown(
        markdown_content, 
        extras=['fenced-code-blocks', 'tables', 'task_list', 'strike']
    )
    
    # Create complete HTML
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Assignment 1 - Housing Price Prediction</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 0.8in;
            }}
            .no-print {{ display: none; }}
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        h3 {{
            color: #2980b9;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        h4 {{
            color: #27ae60;
            margin-top: 15px;
            margin-bottom: 8px;
        }}
        
        p {{
            margin-bottom: 12px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-bottom: 15px;
            padding-left: 25px;
        }}
        
        li {{
            margin-bottom: 5px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 15px 0;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding-left: 15px;
            font-style: italic;
            color: #555;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #95a5a6;
            margin: 30px 0;
        }}
        
        .print-btn {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 20px 0;
        }}
        
        .print-btn:hover {{
            background-color: #2980b9;
        }}
        
        .header-info {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        
        .figure-caption {{
            text-align: center;
            font-style: italic;
            color: #666;
            margin-top: 5px;
            margin-bottom: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="no-print">
        <button class="print-btn" onclick="window.print()">🖨️ Print to PDF</button>
        <p><strong>Instructions:</strong> Click the print button above, then choose "Save as PDF" as your printer destination.</p>
    </div>
    
    <div class="header-info">
        <h1>📊 MLOps Assignment 1 - Housing Price Prediction</h1>
        <p><strong>Complete Project Documentation with Visualizations</strong></p>
        <p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
    
    {html_content}
    
    <div class="no-print">
        <hr>
        <p><strong>End of Document</strong></p>
        <p>To create PDF: Use your browser's print function (Ctrl+P) and select "Save as PDF"</p>
    </div>
</body>
</html>
"""
    
    return full_html

def main():
    """Main function."""
    print("=" * 60)
    print("CREATING HTML DOCUMENTATION WITH EMBEDDED IMAGES")
    print("=" * 60)
    
    try:
        html_content = create_html_document()
        
        # Save HTML file
        output_file = 'MLOps_Assignment1_Complete_Documentation.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML documentation created: {output_file}")
        print("\nTo create PDF:")
        print("1. Open the HTML file in a web browser")
        print("2. Click the 'Print to PDF' button (or press Ctrl+P)")
        print("3. Choose 'Save as PDF' as destination")
        print("4. Adjust print settings if needed")
        print("5. Save the PDF file")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
import markdown

with open("report/combined_report.md", "r", encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Credit Card Fraud Detection - A2 Journal</title>
<style>
body {{
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.6;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    color: #222;
}}
h1 {{ font-size: 28px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
h2 {{ font-size: 22px; margin-top: 32px; }}
h3 {{ font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
code {{ font-family: Consolas, monospace; background-color: #f4f4f4; padding: 2px 4px; }}
pre {{ background-color: #f4f4f4; padding: 12px; overflow-x: auto; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open("report/combined_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("HTML generated.")

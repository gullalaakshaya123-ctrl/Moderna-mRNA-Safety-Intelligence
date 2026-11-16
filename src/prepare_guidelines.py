
from docx import Document

# Read the supplied guidelines document
input_file = "data/Moderna_mRNA_Safety_Guidelines (1).docx"

document = Document(input_file)

# Extract document paragraphs
text_parts = []

for paragraph in document.paragraphs:
    if paragraph.text.strip():
        text_parts.append(paragraph.text.strip())

# Extract table content
for table in document.tables:
    for row in table.rows:
        row_text = " | ".join(
            cell.text.strip()
            for cell in row.cells
        )
        if row_text.strip():
            text_parts.append(row_text)

# Save as plain text
output_file = "data/mrna_safety_guidelines.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n\n".join(text_parts))

print("Guidelines converted successfully!")
print("Output file:", output_file)
print("Total paragraphs and table rows:", len(text_parts))
print("\nPreview:")
print("\n".join(text_parts[:8]))

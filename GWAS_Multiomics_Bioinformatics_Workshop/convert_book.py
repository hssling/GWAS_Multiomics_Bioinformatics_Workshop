#!/usr/bin/env python3
"""
Convert GWAS Multiomics Bioinformatics Textbook from Markdown to PDF and DOCX formats
"""
import os
import pypandoc
from pathlib import Path

def convert_to_formats(input_md, output_dir):
    """
    Convert markdown file to PDF and DOCX formats

    Parameters:
    input_md (str): Path to input markdown file
    output_dir (str): Output directory for converted files
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Base name for output files (without extension)
    base_name = Path(input_md).stem
    pdf_output = os.path.join(output_dir, f"{base_name}.pdf")
    docx_output = os.path.join(output_dir, f"{base_name}.docx")

    # Conversion arguments
    extra_args = [
        '--pdf-engine=pdflatex',
        '--variable', 'geometry:margin=1in',
        '--variable', 'fontsize=11pt',
        '--variable', 'colorlinks=true',
        '--variable', 'linkcolor=blue',
        '--variable', 'urlcolor=blue',
        '--variable', 'citecolor=blue',
        '--variable', 'toc-depth=3',
        '--toc'
    ]

    try:
        print("Converting to PDF...")
        # Convert to PDF
        pypandoc.convert_file(
            input_md,
            'pdf',
            outputfile=pdf_output,
            extra_args=extra_args
        )
        print(f"PDF created: {pdf_output}")

    except Exception as e:
        print(f"PDF conversion failed: {e}")
        print("Make sure LaTeX is installed for PDF conversion")

    try:
        print("Converting to DOCX...")
        # Convert to DOCX
        pypandoc.convert_file(
            input_md,
            'docx',
            outputfile=docx_output
        )
        print(f"DOCX created: {docx_output}")

    except Exception as e:
        print(f"DOCX conversion failed: {e}")

    return pdf_output, docx_output

def main():
    # Input and output paths
    input_md = "GWAS_Multiomics_Bioinformatics_Workshop/Books/GWAS_Multiomics_Bioinformatics_Textbook.md"
    output_dir = "GWAS_Multiomics_Bioinformatics_Workshop/Books"

    if not os.path.exists(input_md):
        print(f"Input file not found: {input_md}")
        return

    print(f"Converting {input_md} to PDF and DOCX...")
    pdf_file, docx_file = convert_to_formats(input_md, output_dir)

    # Check if files were created
    if os.path.exists(pdf_file):
        print(f"✓ PDF file created successfully: {pdf_file}")
    else:
        print(f"✗ PDF file not created: {pdf_file}")

    if os.path.exists(docx_file):
        print(f"✓ DOCX file created successfully: {docx_file}")
    else:
        print(f"✗ DOCX file not created: {docx_file}")

if __name__ == "__main__":
    main()

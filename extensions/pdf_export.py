import os
import json
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

class PDFExportExtension:
    """
    Extension module to export the multi-step study plan to a readable PDF/TXT output.
    Fits seamlessly within the final Streamlit, Chat, and PDF distribution layer.
    """
    
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            
    def export_study_plan(self, student_id: str, student_name: str, study_plan_md: str, final_report_json: dict) -> tuple[str, str]:
        safe_name = "".join([c for c in f"{student_name}_{student_id}" if c.isalpha() or c.isdigit() or c in" -_"]).rstrip()
        file_path_md = os.path.join(self.export_dir, f"Study_Plan_{safe_name}.md")
        file_path_pdf = os.path.join(self.export_dir, f"Study_Plan_{safe_name}.pdf")
        
        with open(file_path_md, "w", encoding="utf-8") as f:
            f.write(f"# Personalized Study Plan for {student_name} ({student_id})\n\n")
            f.write(study_plan_md)

        if HAS_FPDF:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, f"Personalized Study Plan: {student_name} ({student_id})", new_x="LMARGIN", new_y="NEXT", align='C')
            pdf.ln(10)
            
            pdf.set_font('helvetica', '', 11)
            lines = study_plan_md.replace('**', '').replace('##', '').replace('#', '').split('\n')
            for line in lines:
                line_safe = line.encode('ascii', 'ignore').decode('ascii')
                if line_safe.strip():
                    words = line_safe.split(" ")
                    safe_words = []
                    for w in words:
                        if len(w) > 80:
                            safe_words.append(' '.join([w[i:i+80] for i in range(0, len(w), 80)]))
                        else:
                            safe_words.append(w)
                    line_safe = " ".join(safe_words)
                    
                    try:
                        pdf.multi_cell(0, 7, text=line_safe)
                    except Exception:
                        pass
            
            pdf.output(file_path_pdf)
            return file_path_pdf, file_path_md
        else:
            return file_path_md, file_path_md

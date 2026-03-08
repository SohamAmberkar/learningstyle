from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_prompt_pdf():
    output_path = r"C:\Users\manas\Downloads\Project_Prompt.pdf"
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom Style
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    
    normal_style = styles['Normal']
    normal_style.fontSize = 12
    normal_style.leading = 14

    bold_style = ParagraphStyle('Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12)

    # Content
    content = [
        ("Master Project Prompt", title_style),
        ("Act as an expert in Educational Data Mining and AI-driven Adaptive Learning Systems.", normal_style),
        ("", normal_style),
        ("Project Title: AI-Driven Learning Style Prediction for Adaptive E-Learning Systems", bold_style),
        ("Team Members: Manas Dutt (22BCE1350), Urvi Samirkumar Shah (22BCE1174), Soham Yogesh Amberkar (22BCE1770)", normal_style),
        ("Guide: Dr. Vijayalakshmi A", normal_style),
        ("", normal_style),
        ("Project Overview:", bold_style),
        ("We are building a system to automatically predict student learning styles in real-time using Least Mean Squares (LMS) log data, without relying on static, manual surveys.", normal_style),
        ("", normal_style),
        ("1. The Problem:", bold_style),
        ("• Static & Biased: Traditional identification uses manual questionnaires (like Felder-Silverman) which are lengthy and prone to bias.", normal_style),
        ("• Scalability: Manual surveys are impossible to scale for MOOCs.", normal_style),
        ("• Lack of Explainability: Existing ML models function as black boxes.", normal_style),
        ("", normal_style),
        ("2. The Solution & Objectives:", bold_style),
        ("• Develop an AI system that analyzes behavioral logs (video interactions, quiz attempts).", normal_style),
        ("• Map these behaviors to the Felder-Silverman Learning Style Model (FSLSM).", normal_style),
        ("• Use Semi-Supervised Learning to handle the scarcity of labeled data.", normal_style),
        ("• Provide an Analytics Dashboard for faculty.", normal_style),
        ("", normal_style),
        ("3. Proposed Methodology (Architecture):", bold_style),
        ("• Input: Extract anonymized activity logs from Moodle/LMS.", normal_style),
        ("• Preprocessing: SMOTE for class imbalance.", normal_style),
        ("• Core Models: Hybrid Neural Networks, Graph Representation Learning + Fuzzy C-Means, Ensemble Boosting.", normal_style),
        ("• Semi-Supervised Loop: Self-training with unlabeled data.", normal_style),
        ("", normal_style),
        ("4. Technology Stack:", bold_style),
        ("• Python, Scikit-learn, PyTorch/TensorFlow, LightGBM, CatBoost, SHAP.", normal_style),
        ("", normal_style),
        ("5. Expected Outcomes:", bold_style),
        ("• Achieve >85% prediction accuracy.", normal_style),
        ("• Enable real-time adaptation of course content.", normal_style),
    ]

    for text, style in content:
        if text:
            story.append(Paragraph(text, style))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF generated at: {output_path}")

if __name__ == "__main__":
    create_prompt_pdf()

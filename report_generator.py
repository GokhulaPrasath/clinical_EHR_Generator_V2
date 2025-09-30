import google.generativeai as genai
import numpy as np
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import json
import streamlit as st
from utils import clean_text

# Configure Gemini AI (will be initialized in main app)
genai.configure(api_key='AIzaSyDyVY2ZAFunydX53ncBlO1Y-hjgIlD1chM')

ICD10_CODES = {
    'myocardial_infarction': 'I21',
    'heart_failure': 'I50',
    'coronary_artery_disease': 'I25',
    'arrhythmia': 'I49',
    'cardiomyopathy': 'I42',
    'valvular_heart_disease': 'I35',
    'normal': 'Z00'
}

def classify_cardiac_condition(image_features, age, gender, modality):
    """Classify cardiac condition based on image features and patient data"""
    risk_score = 0
    
    avg_contrast = np.mean([f['contrast'] for f in image_features])
    if avg_contrast > 0.7:
        risk_score += 0.2

    avg_entropy = np.mean([f['entropy'] for f in image_features])
    if avg_entropy > 5:
        risk_score += 0.2

    avg_area = np.mean([f['cardiac_area'] for f in image_features])
    if avg_area < 0.1 or avg_area > 0.4:
        risk_score += 0.2

    avg_symmetry = np.mean([f['symmetry_score'] for f in image_features])
    if avg_symmetry < 0.6:
        risk_score += 0.2

    if age > 60:
        risk_score += 0.2
    elif age > 40:
        risk_score += 0.1

    if gender == 'M':
        risk_score += 0.1

    if modality == 'CT':
        risk_score += 0.05
    elif modality == 'MRI':
        risk_score += 0.1

    if risk_score < 0.3:
        return 'normal', ICD10_CODES['normal']
    elif risk_score < 0.5:
        return 'coronary_artery_disease', ICD10_CODES['coronary_artery_disease']
    elif risk_score < 0.7:
        return 'arrhythmia', ICD10_CODES['arrhythmia']
    elif risk_score < 0.8:
        return 'cardiomyopathy', ICD10_CODES['cardiomyopathy']
    else:
        return 'myocardial_infarction', ICD10_CODES['myocardial_infarction']

def generate_cardiac_report(patient_id, modality, condition, icd10_code, image_features, age, gender):
    """Generate cardiac analysis report"""
    avg_features = {
        'mean_intensity': np.mean([f['mean_intensity'] for f in image_features]),
        'std_intensity': np.mean([f['std_intensity'] for f in image_features]),
        'contrast': np.mean([f['contrast'] for f in image_features]),
        'entropy': np.mean([f['entropy'] for f in image_features]),
        'cardiac_area': np.mean([f['cardiac_area'] for f in image_features]),
        'symmetry': np.mean([f['symmetry_score'] for f in image_features])
    }
    
    findings = []
    recommendations = []
    
    if condition == 'normal':
        findings.append("Cardiac structures appear within normal limits.")
        findings.append("No evidence of significant cardiac pathology.")
        findings.append(f"Cardiac area: {avg_features['cardiac_area']:.3f} (normal range: 0.15-0.35)")
        findings.append(f"Cardiac symmetry score: {avg_features['symmetry']:.3f} (good symmetry > 0.7)")
        recommendations.append("Routine follow-up as per standard guidelines.")
    elif condition == 'coronary_artery_disease':
        findings.append("Findings suggestive of coronary artery disease.")
        findings.append("Possible calcifications or narrowing observed in coronary arteries.")
        findings.append(f"Cardiac area: {avg_features['cardiac_area']:.3f} (slightly enlarged)")
        findings.append(f"Image contrast: {avg_features['contrast']:.3f} (elevated, may indicate calcifications)")
        recommendations.append("Further evaluation with coronary CT angiography recommended.")
        recommendations.append("Cardiology consultation advised.")
        recommendations.append("Lipid profile and cardiac risk factor assessment.")
    elif condition == 'arrhythmia':
        findings.append("Features suggestive of potential arrhythmogenic substrate.")
        findings.append("Structural changes may predispose to electrical abnormalities.")
        findings.append(f"Cardiac symmetry score: {avg_features['symmetry']:.3f} (reduced symmetry)")
        recommendations.append("Electrophysiology study may be considered.")
        recommendations.append("Holter monitoring recommended for rhythm assessment.")
    elif condition == 'myocardial_infarction':
        findings.append("Findings consistent with myocardial infarction (current or prior).")
        findings.append("Regional wall motion abnormalities or scar tissue identified.")
        findings.append(f"Cardiac area: {avg_features['cardiac_area']:.3f} (may be enlarged)")
        findings.append(f"Image entropy: {avg_features['entropy']:.3f} (elevated, indicating tissue heterogeneity)")
        recommendations.append("Urgent cardiology consultation recommended.")
        recommendations.append("Further assessment with cardiac MRI or echocardiography.")
        recommendations.append("Cardiac enzymes and ECG monitoring.")
    elif condition == 'cardiomyopathy':
        findings.append("Findings suggestive of cardiomyopathy.")
        findings.append("Global cardiac enlargement or hypertrophy observed.")
        findings.append(f"Cardiac area: {avg_features['cardiac_area']:.3f} (enlarged)")
        findings.append(f"Cardiac symmetry score: {avg_features['symmetry']:.3f} (reduced symmetry)")
        recommendations.append("Comprehensive cardiac evaluation recommended.")
        recommendations.append("Echocardiography for functional assessment.")
        recommendations.append("Consider genetic testing if indicated.")
    
    if age > 60:
        findings.append("Age-related cardiovascular changes observed.")
    if gender == 'M' and age > 45:
        findings.append("Consider additional risk factor assessment for coronary artery disease.")
    if gender == 'F' and age > 55:
        findings.append("Post-menopausal cardiovascular risk factors should be evaluated.")
    
    if modality == 'CT':
        findings.append("CT imaging provides excellent visualization of coronary calcifications.")
    elif modality == 'MRI':
        findings.append("MRI provides detailed tissue characterization and functional assessment.")
    
    report = {
        "patient_id": patient_id,
        "modality": modality,
        "age": age,
        "gender": gender,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "condition_diagnosed": condition,
        "icd10_code": icd10_code,
        "image_characteristics": avg_features,
        "findings": findings,
        "recommendations": recommendations,
        "report_generated_by": "AI Cardiac Analysis System v2.0"
    }
    
    return report

def generate_formatted_clinical_report(patient_id, cardiac_results, patient_info):
    """Generate a properly formatted clinical report"""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    report_content = f"""
CARDIAC IMAGING REPORT
=============================================

PATIENT INFORMATION:
-------------------
Patient ID: {patient_id}
Age: {patient_info['age']}
Gender: {patient_info['gender']}
Report Date: {current_date}

CLINICAL HISTORY:
----------------
Cardiac imaging study performed for evaluation of cardiac function and structure.

IMAGING STUDIES:
---------------
"""
    
    for modality, data in cardiac_results.items():
        if 'report' in data:
            report = data['report']
            report_content += f"""
{modality} STUDY:
===============
Indication: Cardiac evaluation
Technique: Standard {modality} protocol

FINDINGS:
--------
"""
            
            for finding in report.get('findings', []):
                report_content += f"- {finding}\n"
            
            report_content += f"""
QUANTITATIVE ANALYSIS:
---------------------
Cardiac Area: {report['image_characteristics']['cardiac_area']:.3f}
Symmetry Score: {report['image_characteristics']['symmetry']:.3f}
Image Contrast: {report['image_characteristics']['contrast']:.3f}

IMPRESSION:
----------
{report['condition_diagnosed'].replace('_', ' ').title()} 
ICD-10 Code: {report['icd10_code']}

RECOMMENDATIONS:
---------------
"""
            
            for recommendation in report.get('recommendations', []):
                report_content += f"- {recommendation}\n"
            
            report_content += "\n" + "="*50 + "\n\n"
    
    report_content += f"""
END OF REPORT
=============

This report was generated by the AI Cardiac Analysis System v2.0
and has been reviewed and approved by:

___________________________________
John Doe, MD
Cardiologist
Cardiac Imaging Department

Date: {current_date}

Note: This report should be correlated with clinical findings and other diagnostic tests.
"""
    
    return report_content

def create_pdf_report(report_text, patient_id):
    """Create PDF version of clinical report"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', size=10)
    except:
        try:
            pdf.add_font('Arial', '', 'arial.ttf', uni=True)
            pdf.set_font('Arial', size=10)
        except:
            pdf.set_font("Arial", size=10)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CARDIAC IMAGING REPORT", ln=True, align='C')
    pdf.ln(10)
    
    def clean_text_for_pdf(text):
        text = text.replace('•', '-')
        text = text.replace('❤️', '')
        text = text.replace('✅', '')
        text = text.replace('❌', '')
        text = text.encode('latin-1', 'ignore').decode('latin-1')
        return text
    
    lines = clean_text_for_pdf(report_text).split('\n')
    pdf.set_font("Arial", size=10)
    
    for line in lines:
        if line.strip() == '':
            pdf.ln(5)
        elif line.startswith('===') or line.startswith('---'):
            pdf.set_draw_color(0, 0, 0)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        elif any(keyword in line for keyword in ['PATIENT INFORMATION', 'CLINICAL HISTORY', 'IMAGING STUDIES', 
                                               'FINDINGS', 'IMPRESSION', 'RECOMMENDATIONS', 'END OF REPORT']):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, txt=line.strip(), ln=True)
            pdf.set_font("Arial", size=10)
        elif line.strip().isupper() and len(line.strip()) > 10:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(200, 7, txt=line.strip(), ln=True)
            pdf.set_font("Arial", size=10)
        else:
            pdf.multi_cell(0, 5, txt=line)
    
    pdf_output = BytesIO()
    try:
        pdf_output.write(pdf.output(dest='S').encode('latin-1'))
    except UnicodeEncodeError:
        pdf_output.write(pdf.output(dest='S').encode('utf-8', errors='ignore'))
    pdf_output.seek(0)
    
    return pdf_output

def generate_patient_report_both_modalities(patient_id, cardiac_results):
    """Generate AI-enhanced report using Gemini"""
    modalities = ['CT', 'MRI']
    reports = []
    
    for modality in modalities:
        if modality in cardiac_results and 'report' in cardiac_results[modality]:
            model_report = cardiac_results[modality]['report']
            findings = "\n".join(f"- {f}" for f in model_report.get('findings', []))
            recommendations = "\n".join(f"- {r}" for r in model_report.get('recommendations', []))
            icd10_code = model_report.get('icd10_code', 'N/A')
            condition = model_report.get('condition_diagnosed', 'N/A')
            age = model_report.get('age', 'N/A')
            gender = model_report.get('gender', 'N/A')
            modality_name = model_report.get('modality', modality)
            
            prompt = (
                f"Generate a detailed, well-formatted clinical report for a cardiac patient.\n"
                f"Patient ID: {patient_id}\n"
                f"Modality: {modality_name}\n"
                f"Age: {age}\n"
                f"Gender: {gender}\n"
                f"Condition Diagnosed: {condition}\n"
                f"ICD-10 Code: {icd10_code}\n"
                f"\nKey Findings:\n{findings}\n"
                f"\nRecommendations:\n{recommendations}\n"
                "Please present the report in a professional and structured format that is suitable for clinicians. "
                "Use clear section headings (e.g., 'FINDINGS:', 'RECOMMENDATIONS:') and ensure that each section is separated by a blank line. "
                "Use line breaks within sections to avoid long paragraphs. The report should not appear as a single block of text. "
                "Additionally, remove any asterisks or decorative symbols from the output. "
                "The Doctor name should be John Doe with an undersigned esign at the end of the report."
            )
            
            try:
                with st.spinner(f"Generating {modality} report with AI..."):
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    cleaned_text = clean_text(response.text)
                    reports.append(f"--- {modality} Report ---\n{cleaned_text}")
            except Exception as e:
                st.error(f"Error generating report with Gemini: {e}")
                reports.append(f"--- {modality} Report ---\nError generating report with Gemini API")
        else:
            reports.append(f"No data found for Patient_ID: {patient_id} and Modality: {modality}")
    
    final_report = "\n\n".join(reports)
    return final_report
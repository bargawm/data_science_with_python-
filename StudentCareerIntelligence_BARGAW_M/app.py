
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Student Career & Performance Intelligence", layout="wide")

st.title("Student Career & Performance Intelligence System")
st.subheader("Created by: BARGAW M")


def compute_performance_trend(marks_series: pd.Series) -> dict:
	if len(marks_series) < 2:
		return {"trend": 0.0, "consistency": 100}
	x = np.arange(len(marks_series))
	coef = np.polyfit(x, marks_series.values, 1)[0]
	trend = float(coef)
	consistency = max(0, 100 - float(np.std(marks_series)))
	return {"trend": trend, "consistency": consistency}


def detect_sudden_drop(marks_series: pd.Series, threshold: float = 15.0) -> bool:
	if len(marks_series) < 2:
		return False
	diffs = np.diff(marks_series)
	return np.any(diffs <= -threshold)


def productivity_score(study_hours: float, sleep_hours: float, screen_hours: float) -> float:
	score = study_hours * 12 + min(sleep_hours, 8) * 5 - screen_hours * 3
	return float(np.clip(score, 0, 100))


def parse_resume(file) -> str:
	fname = file.name.lower()
	try:
		if fname.endswith('.pdf'):
			import pdfplumber
			text = []
			with pdfplumber.open(BytesIO(file.read())) as pdf:
				for p in pdf.pages:
					text.append(p.extract_text() or '')
			return '\n'.join(text)
		elif fname.endswith('.docx') or fname.endswith('.doc'):
			from docx import Document
			doc = Document(BytesIO(file.read()))
			return '\n'.join(p.text for p in doc.paragraphs)
		else:
			# try reading as text
			return file.getvalue().decode('utf-8', errors='ignore')
	except Exception:
		return ''


def extract_skills_from_text(text: str, top_k: int = 10) -> list:
	# Lightweight skill extraction using keyword matching + TF-IDF weighting
	from sklearn.feature_extraction.text import TfidfVectorizer
	skills_lookup = [
		'python','pandas','numpy','scikit-learn','sql','excel','tableau','power bi','machine learning',
		'data analysis','data visualization','statistics','deep learning','nlp','tensorflow','keras',
		'communication','presentation','leadership','git','aws','azure'
	]
	docs = [' '.join(skills_lookup), text]
	vec = TfidfVectorizer(vocabulary=skills_lookup, lowercase=True)
	try:
		X = vec.fit_transform(docs)
		scores = np.ravel(X.toarray()[1])
		skill_scores = list(zip(vec.get_feature_names_out(), scores))
		skill_scores = sorted(skill_scores, key=lambda x: x[1], reverse=True)
		return [s for s,sc in skill_scores if sc>0][:top_k]
	except Exception:
		return []


def predict_role_from_skills(skills: list) -> str:
	# Simple rule-based mapping using a small role profile dictionary
	role_profiles = {
		'Data Scientist': ['python','numpy','pandas','machine learning','nlp','statistics'],
		'Data Analyst': ['excel','sql','tableau','power bi','pandas','data visualization'],
		'ML Engineer': ['python','tensorflow','keras','deep learning','git','aws'],
		'Software Engineer': ['python','git','communication'],
		'Business Analyst': ['excel','communication','presentation','sql']
	}
	if not skills:
		return 'Unknown / Needs stronger resume'
	scores = {}
	for role, prof in role_profiles.items():
		scores[role] = len(set(skills) & set(prof))
	best = max(scores.items(), key=lambda x: x[1])
	return best[0] if best[1] > 0 else 'Generalist / Needs Upskilling'


with st.sidebar:
	st.header('Input Data')
	st.markdown('Upload academic CSV/Excel (columns: semester, marks, attendance%)')
	academic_file = st.file_uploader('Academic data file', type=['csv','xlsx','xls'])
	st.markdown('Upload resume (PDF/DOCX)')
	resume_file = st.file_uploader('Resume file', type=['pdf','docx','doc','txt'])

st.markdown('**Academic Performance Analytics**')
col1, col2 = st.columns([2,1])
with col1:
	if academic_file is not None:
		try:
			if academic_file.name.lower().endswith('.csv'):
				df = pd.read_csv(academic_file)
			else:
				df = pd.read_excel(academic_file)
			st.dataframe(df.head())
			if 'marks' in df.columns:
				marks_series = df['marks'].astype(float)
				perf = compute_performance_trend(marks_series)
				st.metric('Trend (slope per semester)', f"{perf['trend']:.2f}")
				st.metric('Consistency Score', f"{perf['consistency']:.1f}%")
				if detect_sudden_drop(marks_series):
					st.warning('Sudden drop detected in marks! Investigate recent semesters.')
			else:
				st.info('Please provide a column named `marks`.')
		except Exception as e:
			st.error('Failed to read academic file: ' + str(e))
	else:
		st.info('You can provide a CSV/Excel file with semester marks to analyze trends.')

with col2:
	st.markdown('Quick inputs')
	latest_marks = st.number_input('Latest semester marks', 0, 100, 75)
	attendance = st.slider('Attendance percentage', 0, 100, 85)
	st.progress(latest_marks/100)

st.markdown('**Behavioral Analytics Engine**')
bh1, bh2 = st.columns(2)
with bh1:
	study = st.slider('Daily study hours', 0.0, 16.0, 4.0)
	sleep = st.slider('Daily sleep hours', 0.0, 12.0, 7.0)
	screen = st.slider('Daily screen (non-study) hours', 0.0, 16.0, 3.0)
	prod = productivity_score(study, sleep, screen)
	st.metric('Productivity Score', f"{prod:.0f}/100")
with bh2:
	st.markdown('Habit insights')
	st.write(f'Study: {study}h • Sleep: {sleep}h • Screen: {screen}h')
	if prod < 30:
		st.warning('Low productivity — consider increasing focused study time and reducing non-essential screen time')

st.markdown('**Career Intelligence (AI + NLP)**')
if resume_file is not None:
	text = parse_resume(resume_file)
	if not text.strip():
		st.error('Could not extract text from resume. Try a different file type.')
	else:
		skills = extract_skills_from_text(text)
		st.subheader('Extracted Skills')
		if skills:
			st.write(', '.join(skills))
		else:
			st.write('No clear skill keywords found.')
		predicted_role = predict_role_from_skills(skills)
		st.success(f'Predicted Best-fit Role: {predicted_role}')
else:
	st.info('Upload a resume to run resume intelligence. Supported: PDF, DOCX, TXT')

st.markdown('**AI Recommendation & Future Simulator**')
run = st.button('Run Overall Assessment')
if run:
	# Aggregate simple scores into health and readiness
	academic_score = latest_marks * 0.6 + attendance * 0.4
	behavior_score = prod
	readiness = 0.6 * (academic_score/1.0) * 0.6 + 0.4 * behavior_score
	# map into 0-100
	performance_health = float(np.clip((academic_score*0.7 + behavior_score*0.3)/1.5, 0, 100))
	career_readiness = float(np.clip(readiness/1.5, 0, 100))
	st.metric('Performance Health', f"{performance_health:.1f}/100")
	st.metric('Career Readiness', f"{career_readiness:.1f}/100")

	recs = []
	if performance_health < 50:
		recs.append('Focus on fundamentals: revise last semester topics and seek tutoring')
	if behavior_score < 40:
		recs.append('Improve routine: increase focused study, reduce nonessential screen time')
	if resume_file is None or (resume_file is not None and (not skills)):
		recs.append('Enhance resume: add measurable projects, list core technical skills')
	if performance_health > 75 and career_readiness > 60:
		recs.append('Consider internships or applied projects to strengthen portfolio')

	st.subheader('Personalized Recommendations')
	for r in recs:
		st.write('- ' + r)

	st.markdown('**What-if Simulator**')
	st.write('Try changing study hours, sleep, or marks to see predicted scores.')

st.markdown('---')
st.caption('This lightweight system uses simple ML/NLP building blocks for offline analysis and is optimized for low memory usage. Packaging to a single `.exe` can be done using `pyinstaller` (see README).')

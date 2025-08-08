from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle
import logging
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Career Recommendation Comparison API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://careerwiz-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
)

# Global variables
recommender = None
courses_df = None

class XGBoostCareerRecommender:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 objective='reg:squarederror', random_state=42):
        """
        Initialize the XGBoost Career Recommender

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            objective: XGBoost objective function
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.course_encoder = LabelEncoder()
        self.feature_names = []

    def extract_advanced_features(self, students_df: pd.DataFrame, courses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for XGBoost training

        Args:
            students_df: DataFrame with student information
            courses_df: DataFrame with course information

        Returns:
            DataFrame with engineered features
        """
        features_list = []

        # Define subject categories
        all_subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English', 'Geography', 'History']
        stem_subjects = ['Mathematics', 'Physics', 'Chemistry', 'Biology']
        humanities_subjects = ['English', 'Geography', 'History']

        # Define interest categories
        interest_categories = {
            'technology': ['programming', 'computers', 'innovation', 'AI', 'software', 'data', 'machine learning', 'robotics'],
            'healthcare': ['medicine', 'nursing', 'biology', 'helping', 'health', 'medical', 'care'],
            'business': ['entrepreneurship', 'marketing', 'finance', 'management', 'economics', 'business'],
            'creative': ['art', 'design', 'music', 'writing', 'creative', 'media', 'storytelling'],
            'social': ['teaching', 'counseling', 'social work', 'psychology', 'education', 'helping people'],
            'research': ['research', 'analysis', 'investigation', 'study', 'academic'],
            'communication': ['communication', 'public speaking', 'presentation', 'media', 'journalism']
        }

        for index in students_df.index:
            student = students_df.loc[index]

            if 'subjects' not in student:
                logger.error(f"Error: 'subjects' not in student data for index {index}. Columns: {student.index.tolist()}")
                continue # Skip this student if 'subjects' is missing

            student_subjects = set(student['subjects'].split(', '))
            student_interests = set(student['interests'].split(', '))

            for _, course in courses_df.iterrows():
                course_subjects = set(course['subjects'].split(', '))
                course_skills = set(course['skills'].split(', '))

                features = {}

                # Student ID and Course ID (for reference)
                features['student_id'] = student['student_id']
                features['course_name'] = course['course_name']

                # Basic subject matching features
                for subject in all_subjects:
                    features[f'student_has_{subject.lower()}'] = 1 if subject in student_subjects else 0
                    features[f'course_requires_{subject.lower()}'] = 1 if subject in course_subjects else 0

                # Subject overlap features
                features['subject_overlap_count'] = len(student_subjects.intersection(course_subjects))
                features['subject_overlap_ratio'] = (len(student_subjects.intersection(course_subjects)) /
                                                   len(student_subjects.union(course_subjects)) if student_subjects.union(course_subjects) else 0)

                # Interest-skill alignment features
                features['interest_skill_overlap'] = len(student_interests.intersection(course_skills))
                features['interest_skill_ratio'] = (len(student_interests.intersection(course_skills)) /
                                                   len(student_interests.union(course_skills)) if student_interests.union(course_skills) else 0)

                # Subject category features
                student_stem_count = sum(1 for subj in student_subjects if subj in stem_subjects)
                student_humanities_count = sum(1 for subj in student_subjects if subj in humanities_subjects)

                course_stem_count = sum(1 for subj in course_subjects if subj in stem_subjects)
                course_humanities_count = sum(1 for subj in course_subjects if subj in humanities_subjects)

                features['student_stem_ratio'] = student_stem_count / len(stem_subjects) if len(stem_subjects) > 0 else 0
                features['student_humanities_ratio'] = student_humanities_count / len(humanities_subjects) if len(humanities_subjects) > 0 else 0
                features['course_stem_ratio'] = course_stem_count / len(stem_subjects) if len(stem_subjects) > 0 else 0
                features['course_humanities_ratio'] = course_humanities_count / len(humanities_subjects) if len(humanities_subjects) > 0 else 0


                # Alignment between student and course preferences
                features['stem_alignment'] = min(features['student_stem_ratio'], features['course_stem_ratio'])
                features['humanities_alignment'] = min(features['student_humanities_ratio'], features['course_humanities_ratio'])

                # Interest category features
                for category, keywords in interest_categories.items():
                    student_category_score = sum(1 for interest in student_interests
                                               if any(keyword.lower() in interest.lower() for keyword in keywords))
                    course_category_score = sum(1 for skill in course_skills
                                              if any(keyword.lower() in skill.lower() for keyword in keywords))

                    features[f'student_{category}_interest'] = student_category_score
                    features[f'course_{category}_relevance'] = course_category_score
                    features[f'{category}_alignment'] = min(student_category_score, course_category_score)

                # Diversity features
                features['student_subject_diversity'] = len(student_subjects)
                features['student_interest_diversity'] = len(student_interests)
                features['course_subject_breadth'] = len(course_subjects)
                features['course_skill_breadth'] = len(course_skills)

                # Target variable: compatibility score
                subject_score = len(student_subjects.intersection(course_subjects))
                interest_score = len(student_interests.intersection(course_skills))
                total_possible = len(student_subjects) + len(student_interests)

                features['compatibility_score'] = (subject_score + interest_score) / total_possible if total_possible > 0 else 0

                features_list.append(features)

        return pd.DataFrame(features_list)

    def predict_for_student(self, student_data: Dict, courses_df: pd.DataFrame, top_k=5) -> List[Tuple[str, float]]:
        """
        Predict career recommendations for a single student

        Args:
            student_data: Dictionary with 'subjects' and 'interests' keys
            courses_df: DataFrame with course information
            top_k: Number of top recommendations to return
        Returns:
            List of (course_name, confidence_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Create a temporary DataFrame for the student
        temp_student_df = pd.DataFrame([{
            'student_id': 'temp_student',
            'subjects': ', '.join(student_data['subjects']),
            'interests': ', '.join(student_data['interests'])
        }])

        # Extract features for all student-course combinations
        features_df = self.extract_advanced_features(temp_student_df, courses_df)

        # Prepare features for prediction
        feature_cols = [col for col in features_df.columns
                       if col not in ['student_id', 'course_name', 'compatibility_score']]

        # Ensure all expected features are present, fill missing with 0
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        # Reorder columns to match training data
        X = features_df[self.feature_names]

        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Combine with course names
        recommendations = []
        for i, course_name in enumerate(features_df['course_name']):
            recommendations.append((course_name, float(predictions[i])))

        # Sort by prediction score and return top K
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

    def load_model(self, filepath: str):
        """Load a trained model and preprocessors"""
        try:
            # Load preprocessors and metadata first
            with open(f"{filepath}_preprocessors.pkl", 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.course_encoder = data['course_encoder']
                self.feature_names = data['feature_names']
                self.n_estimators = data['n_estimators']
                self.max_depth = data['max_depth']
                self.learning_rate = data['learning_rate']
                self.objective = data['objective']
                self.random_state = data['random_state']

            # Initialize the model with loaded hyperparameters before loading the model state
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                random_state=self.random_state
            )
            # Load XGBoost model state
            self.model.load_model(f"{filepath}_xgboost.json")

            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise FileNotFoundError(f"Model file not found: {e}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")


# Pydantic model for request body
class StudentInput(BaseModel):
    subjects: List[str]
    interests: List[str]
    top_k: Optional[int] = 5 # Default to 5 recommendations

# Event handler to load the model and data on startup
@app.on_event("startup")
async def load_model_and_data():
    global recommender, courses_df
    logger.info("Loading model and data...")
    try:
        recommender = XGBoostCareerRecommender()
        recommender.load_model('./xgboost_career_model')
        courses_df = pd.read_csv('./Courses.csv')
        logger.info("Model and data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model or data: {e}")
        recommender = None # Ensure recommender is None if loading fails
        courses_df = None


# Prediction endpoint
@app.post("/predict")
async def predict_career(student_input: StudentInput):
    global recommender, courses_df

    if recommender is None or courses_df is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded. Please check server logs.")

    student_data = {
        'subjects': student_input.subjects,
        'interests': student_input.interests
    }

    try:
        recommendations = recommender.predict_for_student(
            student_data,
            courses_df,
            top_k=student_input.top_k
        )
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if recommender is not None and courses_df is not None:
        return {"status": "ok", "model_loaded": True, "data_loaded": True}
    else:
        return {"status": "warning", "model_loaded": recommender is not None, "data_loaded": courses_df is not None, "message": "Model or data not fully loaded"}

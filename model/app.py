import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import logging
from typing import List, Dict, Tuple
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the XGBoostCareerRecommender class (copy the class definition from your notebook)
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

                features['student_stem_ratio'] = student_stem_count / len(stem_subjects)
                features['student_humanities_ratio'] = student_humanities_count / len(humanities_subjects)
                features['course_stem_ratio'] = course_stem_count / len(stem_subjects)
                features['course_humanities_ratio'] = course_humanities_count / len(humanities_subjects)

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


    def prepare_training_data(self, students_df: pd.DataFrame, courses_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for XGBoost

        Args:
            students_df: DataFrame with student information
            courses_df: DataFrame with course information

        Returns:
            Tuple of (features, targets)
        """
        logger.info("Extracting advanced features...")

        # Extract features
        features_df = self.extract_advanced_features(students_df, courses_df)

        # Separate features and target
        target_col = 'compatibility_score'
        feature_cols = [col for col in features_df.columns
                       if col not in ['student_id', 'course_name', target_col]]

        X = features_df[feature_cols]
        y = features_df[target_col]

        # Store feature names
        self.feature_names = feature_cols

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Prepared {X_scaled.shape[0]} training samples with {X_scaled.shape[1]} features")

        return X_scaled, y.values

    def train(self, students_df: pd.DataFrame, courses_df: pd.DataFrame,
              test_size=0.2, early_stopping_rounds=20):
        """
        Train the XGBoost model

        Args:
            students_df: DataFrame with student information
            courses_df: DataFrame with course information
            test_size: Fraction of data for testing
            early_stopping_rounds: Early stopping patience
        """
        logger.info("Preparing training data...")

        # Prepare training data
        X, y = self.prepare_training_data(students_df, courses_df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )

        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.random_state,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            subsample=0.8,   # Subsample ratio
            colsample_bytree=0.8,  # Feature sampling
            eval_metric='rmse'
        )

        # Train model without early stopping callbacks to resolve TypeError
        logger.info("Training XGBoost model...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        logger.info(f"Training RMSE: {train_rmse:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")

        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("Training completed!")

        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': importance_df
        }

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

        X = features_df[feature_cols]
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

    def get_feature_importance(self, top_n=20) -> pd.DataFrame:
        """
        Get feature importance from trained model

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save!")

        # Save XGBoost model
        self.model.save_model(f"{filepath}_xgboost.json")

        # Save preprocessors and metadata
        with open(f"{filepath}_preprocessors.pkl", 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'course_encoder': self.course_encoder,
                'feature_names': self.feature_names,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'objective': self.objective,
                'random_state': self.random_state
            }, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and preprocessors"""
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

# --- Streamlit App Code ---

st.title("Career Recommendation System")

# Load the trained model and data
@st.cache_resource
def load_recommender_and_data():
    recommender = XGBoostCareerRecommender()
    try:
        recommender.load_model('xgboost_career_model')
        courses_df = pd.read_csv('./sample_data/Courses.csv')
        return recommender, courses_df
    except FileNotFoundError:
        st.error("Model files not found. Please run the training code first.")
        return None, None

recommender, courses_df = load_recommender_and_data()

if recommender and courses_df is not None:
    st.write("Enter your academic subjects and interests to get career recommendations.")

    # Get user input
    subjects_input = st.text_input("Enter your subjects (comma-separated, e.g., Mathematics, Physics, English)")
    interests_input = st.text_input("Enter your interests (comma-separated, e.g., Programming, AI, Data Science)")

    if st.button("Get Recommendations"):
        if subjects_input and interests_input:
            student_subjects = [s.strip() for s in subjects_input.split(',')]
            student_interests = [i.strip() for i in interests_input.split(',')]

            student_data = {
                'subjects': student_subjects,
                'interests': student_interests
            }

            with st.spinner("Getting recommendations..."):
                recommendations = recommender.predict_for_student(student_data, courses_df)

            st.subheader("Top Career Recommendations:")
            if recommendations:
                for course, confidence in recommendations:
                    st.write(f"- {course} (Confidence: {confidence:.3f})")
            else:
                st.info("No recommendations found based on your input.")
        else:
            st.warning("Please enter both subjects and interests.")

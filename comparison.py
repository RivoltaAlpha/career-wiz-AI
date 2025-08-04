from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import time
import os

# Import your custom models
from deep_learning_recommender import DeepLearningCareerRecommender
from xgboost_recommender import XGBoostCareerRecommender

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
deep_learning_model = None
xgboost_model = None
students_df = None
courses_df = None
models_trained = False

class StudentData(BaseModel):
    subjects: List[str]
    interests: List[str]

class ModelChoice(BaseModel):
    model_type: str  # "deep_learning", "xgboost", or "both"

class ComparisonRequest(BaseModel):
    subjects: List[str]
    interests: List[str]
    top_k: Optional[int] = 5

class RecommendationResponse(BaseModel):
    model_name: str
    recommendations: List[Dict[str, float]]
    prediction_time: float
    confidence_scores: List[float]

class ComparisonResponse(BaseModel):
    deep_learning: Optional[RecommendationResponse]
    xgboost: Optional[RecommendationResponse]
    comparison_summary: Dict

def load_data():
    """Load student and course data"""
    global students_df, courses_df
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        students_path = os.path.join(base_dir, 'student_data.csv')
        courses_path = os.path.join(base_dir, 'Courses.csv')
        
        students_df = pd.read_csv(students_path)
        courses_df = pd.read_csv(courses_path)
        
        logger.info("Data loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

def train_models():
    """Train both models"""
    global deep_learning_model, xgboost_model, models_trained
    
    try:
        # Initialize models
        deep_learning_model = DeepLearningCareerRecommender(
            embedding_dim=64,
            hidden_layers=[128, 64]
        )
        
        xgboost_model = XGBoostCareerRecommender(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        # Train Deep Learning model
        logger.info("Training Deep Learning model...")
        deep_learning_model.train(students_df, courses_df, epochs=30, batch_size=64)
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        xgboost_model.train(students_df, courses_df)
        
        models_trained = True
        logger.info("Both models trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        models_trained = False
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up application...")
    
    if not load_data():
        raise HTTPException(status_code=500, detail="Failed to load data")
    
    if not train_models():
        raise HTTPException(status_code=500, detail="Failed to train models")
    
    logger.info("Application startup complete")

@app.get("/")
def read_root():
    return {"message": "Career Recommendation Comparison API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_trained": models_trained,
        "data_loaded": students_df is not None and courses_df is not None
    }

@app.get("/models/status")
async def get_model_status():
    """Get status of trained models"""
    return {
        "deep_learning_trained": deep_learning_model is not None,
        "xgboost_trained": xgboost_model is not None,
        "models_ready": models_trained
    }

@app.post("/predict/deep_learning")
async def predict_deep_learning(student_data: StudentData):
    """Get predictions from Deep Learning model only"""
    if not models_trained or deep_learning_model is None:
        raise HTTPException(status_code=503, detail="Deep Learning model not available")
    
    try:
        start_time = time.time()
        
        student_dict = {
            'subjects': student_data.subjects,
            'interests': student_data.interests
        }
        
        recommendations = deep_learning_model.predict_for_student(
            student_dict, courses_df, top_k=5
        )
        
        prediction_time = time.time() - start_time
        
        formatted_recommendations = [
            {"course_name": course, "confidence": float(score)}
            for course, score in recommendations
        ]
        
        return {
            "model_name": "Deep Learning",
            "recommendations": formatted_recommendations,
            "prediction_time": prediction_time,
            "total_courses_considered": len(courses_df)
        }
        
    except Exception as e:
        logger.error(f"Error in Deep Learning prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating Deep Learning recommendations")

@app.post("/predict/xgboost")
async def predict_xgboost(student_data: StudentData):
    """Get predictions from XGBoost model only"""
    if not models_trained or xgboost_model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not available")
    
    try:
        start_time = time.time()
        
        student_dict = {
            'subjects': student_data.subjects,
            'interests': student_data.interests
        }
        
        recommendations = xgboost_model.predict_for_student(
            student_dict, courses_df, top_k=5
        )
        
        prediction_time = time.time() - start_time
        
        formatted_recommendations = [
            {"course_name": course, "confidence": float(score)}
            for course, score in recommendations
        ]
        
        return {
            "model_name": "XGBoost",
            "recommendations": formatted_recommendations,
            "prediction_time": prediction_time,
            "total_courses_considered": len(courses_df)
        }
        
    except Exception as e:
        logger.error(f"Error in XGBoost prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating XGBoost recommendations")

@app.post("/predict/compare")
async def compare_models(request: ComparisonRequest):
    """Compare predictions from both models"""
    if not models_trained:
        raise HTTPException(status_code=503, detail="Models not trained")
    
    student_dict = {
        'subjects': request.subjects,
        'interests': request.interests
    }
    
    results = {}
    
    # Get Deep Learning predictions
    if deep_learning_model is not None:
        try:
            start_time = time.time()
            dl_recommendations = deep_learning_model.predict_for_student(
                student_dict, courses_df, top_k=request.top_k
            )
            dl_time = time.time() - start_time
            
            results['deep_learning'] = {
                "model_name": "Deep Learning",
                "recommendations": [
                    {"course_name": course, "confidence": float(score)}
                    for course, score in dl_recommendations
                ],
                "prediction_time": dl_time,
                "confidence_scores": [float(score) for _, score in dl_recommendations]
            }
        except Exception as e:
            logger.error(f"Error in Deep Learning prediction: {str(e)}")
            results['deep_learning'] = {"error": str(e)}
    
    # Get XGBoost predictions
    if xgboost_model is not None:
        try:
            start_time = time.time()
            xgb_recommendations = xgboost_model.predict_for_student(
                student_dict, courses_df, top_k=request.top_k
            )
            xgb_time = time.time() - start_time
            
            results['xgboost'] = {
                "model_name": "XGBoost",
                "recommendations": [
                    {"course_name": course, "confidence": float(score)}
                    for course, score in xgb_recommendations
                ],
                "prediction_time": xgb_time,
                "confidence_scores": [float(score) for _, score in xgb_recommendations]
            }
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {str(e)}")
            results['xgboost'] = {"error": str(e)}
    
    # Generate comparison summary
    comparison_summary = generate_comparison_summary(results)
    
    return {
        "deep_learning": results.get('deep_learning'),
        "xgboost": results.get('xgboost'),
        "comparison_summary": comparison_summary
    }

def generate_comparison_summary(results: Dict) -> Dict:
    """Generate a summary comparing the two models' outputs"""
    summary = {}
    
    dl_result = results.get('deep_learning', {})
    xgb_result = results.get('xgboost', {})
    
    # Performance comparison
    if 'prediction_time' in dl_result and 'prediction_time' in xgb_result:
        summary['speed_comparison'] = {
            'deep_learning_time': dl_result['prediction_time'],
            'xgboost_time': xgb_result['prediction_time'],
            'faster_model': 'deep_learning' if dl_result['prediction_time'] < xgb_result['prediction_time'] else 'xgboost'
        }
    
    # Confidence comparison
    if 'confidence_scores' in dl_result and 'confidence_scores' in xgb_result:
        dl_avg_confidence = np.mean(dl_result['confidence_scores'])
        xgb_avg_confidence = np.mean(xgb_result['confidence_scores'])
        
        summary['confidence_comparison'] = {
            'deep_learning_avg_confidence': float(dl_avg_confidence),
            'xgboost_avg_confidence': float(xgb_avg_confidence),
            'more_confident_model': 'deep_learning' if dl_avg_confidence > xgb_avg_confidence else 'xgboost'
        }
    
    # Recommendation overlap
    if 'recommendations' in dl_result and 'recommendations' in xgb_result:
        dl_courses = set(rec['course_name'] for rec in dl_result['recommendations'])
        xgb_courses = set(rec['course_name'] for rec in xgb_result['recommendations'])
        
        overlap = len(dl_courses.intersection(xgb_courses))
        total_recommendations = len(dl_courses.union(xgb_courses))
        summary['recommendation_overlap'] = {
            'overlap_count': overlap,
            'total_unique_recommendations': total_recommendations,
            'overlap_percentage': (overlap / total_recommendations) * 100 if total_recommendations > 0 else 0
        }
    else:
        summary['recommendation_overlap'] = {
            'overlap_count': 0,
            'total_unique_recommendations': 0,
            'overlap_percentage': 0
        }

        logger.info(f"Recommendation overlap: {summary['recommendation_overlap']}")
    return summary


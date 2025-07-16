import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
import logging

# Import your custom recommenders
from deep_learning_recommender import DeepLearningCareerRecommender
from xgboost_recommender import XGBoostCareerRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, students_df: pd.DataFrame, courses_df: pd.DataFrame):
        """
        Initialize the model comparison framework
        
        Args:
            students_df: DataFrame with student information
            courses_df: DataFrame with course information
        """
        self.students_df = students_df
        self.courses_df = courses_df
        self.results = {}
        
    def evaluate_model_performance(self, model, model_name: str) -> Dict:
        """
        Evaluate a model's performance using various metrics
        
        Args:
            model: Trained model instance
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Evaluating {model_name} model...")
        
        # Test on a sample of students
        test_students = self.students_df.sample(min(100, len(self.students_df)), random_state=42)
        
        predictions = []
        actual_scores = []
        recommendation_times = []
        
        for _, student in test_students.iterrows():
            student_data = {
                'subjects': student['subjects'].split(', '),
                'interests': student['interests'].split(', ')
            }
            
            # Measure prediction time
            start_time = time.time()
            try:
                recommendations = model.predict_for_student(student_data, self.courses_df, top_k=10)
                prediction_time = time.time() - start_time
                recommendation_times.append(prediction_time)
                
                # Calculate actual compatibility scores for comparison
                for course_name, predicted_score in recommendations:
                    course_row = self.courses_df[self.courses_df['course_name'] == course_name].iloc[0]
                    actual_score = self.calculate_actual_compatibility(student_data, course_row)
                    
                    predictions.append(predicted_score)
                    actual_scores.append(actual_score)
                    
            except Exception as e:
                logger.error(f"Error predicting for student {student['student_id']}: {str(e)}")
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            mse = mean_squared_error(actual_scores, predictions)
            mae = mean_absolute_error(actual_scores, predictions)
            r2 = r2_score(actual_scores, predictions)
            
            # Calculate ranking metrics
            ndcg = self.calculate_ndcg(actual_scores, predictions)
            map_score = self.calculate_map(actual_scores, predictions)
            
            avg_prediction_time = np.mean(recommendation_times)
            
            return {
                'model_name': model_name,
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'ndcg': ndcg,
                'map_score': map_score,
                'avg_prediction_time': avg_prediction_time,
                'total_predictions': len(predictions)
            }
        else:
            return {
                'model_name': model_name,
                'error': 'No successful predictions made'
            }
    
    def calculate_actual_compatibility(self, student_data: Dict, course_row: pd.Series) -> float:
        """
        Calculate actual compatibility score between student and course
        
        Args:
            student_data: Student information
            course_row: Course information
            
        Returns:
            Compatibility score between 0 and 1
        """
        student_subjects = set(student_data['subjects'])
        student_interests = set(student_data['interests'])
        
        course_subjects = set(course_row['subjects'].split(', '))
        course_skills = set(course_row['skills'].split(', '))
        
        # Calculate overlap
        subject_overlap = len(student_subjects.intersection(course_subjects))
        interest_overlap = len(student_interests.intersection(course_skills))
        
        total_possible = len(student_subjects) + len(student_interests)
        
        if total_possible == 0:
            return 0.0
        
        return (subject_overlap + interest_overlap) / total_possible
    
    def calculate_ndcg(self, actual_scores: List[float], predicted_scores: List[float], k=5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Args:
            actual_scores: List of actual relevance scores
            predicted_scores: List of predicted scores
            k: Number of top items to consider
            
        Returns:
            NDCG score
        """
        if len(actual_scores) == 0:
            return 0.0
        
        # Sort by predicted scores
        sorted_indices = np.argsort(predicted_scores)[::-1][:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, idx in enumerate(sorted_indices):
            if i < len(actual_scores):
                dcg += (2 ** actual_scores[idx] - 1) / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        ideal_indices = np.argsort(actual_scores)[::-1][:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            if i < len(actual_scores):
                idcg += (2 ** actual_scores[idx] - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_map(self, actual_scores: List[float], predicted_scores: List[float]) -> float:
        """
        Calculate Mean Average Precision
        
        Args:
            actual_scores: List of actual relevance scores
            predicted_scores: List of predicted scores
            
        Returns:
            MAP score
        """
        if len(actual_scores) == 0:
            return 0.0
        
        # Convert to binary relevance (threshold = 0.5)
        binary_actual = [1 if score > 0.5 else 0 for score in actual_scores]
        
        # Sort by predicted scores
        sorted_indices = np.argsort(predicted_scores)[::-1]
        
        # Calculate AP
        relevant_count = 0
        precision_sum = 0.0
        
        for i, idx in enumerate(sorted_indices):
            if binary_actual[idx] == 1:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / relevant_count
    
    def evaluate_recommendation_diversity(self, model, model_name: str) -> Dict:
        """
        Evaluate diversity of recommendations
        
        Args:
            model: Trained model instance
            model_name: Name of the model
            
        Returns:
            Dictionary with diversity metrics
        """
        logger.info(f"Evaluating recommendation diversity for {model_name}...")
        
        all_recommendations = []
        category_distributions = []
        
        # Define course categories
        course_categories = {}
        for _, course in self.courses_df.iterrows():
            subjects = course['subjects'].split(', ')
            if any(subj in ['Mathematics', 'Physics', 'Chemistry', 'Biology'] for subj in subjects):
                course_categories[course['course_name']] = 'STEM'
            elif any(subj in ['English', 'Literature', 'History', 'Geography'] for subj in subjects):
                course_categories[course['course_name']] = 'Humanities'
            else:
                course_categories[course['course_name']] = 'Other'
        
        # Get recommendations for a sample of students
        sample_students = self.students_df.sample(min(50, len(self.students_df)), random_state=42)
        
        for _, student in sample_students.iterrows():
            student_data = {
                'subjects': student['subjects'].split(', '),
                'interests': student['interests'].split(', ')
            }
            
            try:
                recommendations = model.predict_for_student(student_data, self.courses_df, top_k=5)
                recommended_courses = [course_name for course_name, _ in recommendations]
                all_recommendations.extend(recommended_courses)
                
                # Calculate category distribution for this student
                categories = [course_categories.get(course, 'Other') for course in recommended_courses]
                category_dist = {
                    'STEM': categories.count('STEM') / len(categories),
                    'Humanities': categories.count('Humanities') / len(categories),
                    'Other': categories.count('Other') / len(categories)
                }
                category_distributions.append(category_dist)
                
            except Exception as e:
                logger.error(f"Error getting recommendations for student {student['student_id']}: {str(e)}")
                continue
        
        # Calculate diversity metrics
        unique_courses = len(set(all_recommendations))
        total_courses = len(self.courses_df)
        coverage = unique_courses / total_courses
        
        # Calculate average category diversity
        avg_category_diversity = np.mean([
            len([v for v in dist.values() if v > 0]) for dist in category_distributions
        ])
        
        return {
            'model_name': model_name,
            'unique_courses_recommended': unique_courses,
            'total_courses': total_courses,
            'coverage': coverage,
            'avg_category_diversity': avg_category_diversity,
            'category_distributions': category_distributions
        }
    
    def run_comparison(self) -> Dict:
        """
        Run comprehensive comparison between models
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting model comparison...")
        
        # Initialize models
        deep_learning_model = DeepLearningCareerRecommender(
            embedding_dim=64,  # Smaller for faster training
            hidden_layers=[128, 64]
        )
        
        xgboost_model = XGBoostCareerRecommender(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        # Train models
        logger.info("Training Deep Learning model...")
        dl_start_time = time.time()
        try:
            deep_learning_model.train(self.students_df, self.courses_df, epochs=30, batch_size=64)
            dl_training_time = time.time() - dl_start_time
            dl_trained = True
        except Exception as e:
            logger.error(f"Error training Deep Learning model: {str(e)}")
            dl_training_time = 0
            dl_trained = False
        
        logger.info("Training XGBoost model...")
        xgb_start_time = time.time()
        try:
            xgboost_model.train(self.students_df, self.courses_df)
            xgb_training_time = time.time() - xgb_start_time
            xgb_trained = True
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            xgb_training_time = 0
            xgb_trained = False
        
        # Evaluate models
        results = {
            'training_times': {
                'deep_learning': dl_training_time,
                'xgboost': xgb_training_time
            },
            'performance_metrics': {},
            'diversity_metrics': {}
        }
        
        if dl_trained:
            dl_performance = self.evaluate_model_performance(deep_learning_model, 'Deep Learning')
            dl_diversity = self.evaluate_recommendation_diversity(deep_learning_model, 'Deep Learning')
            
            results['performance_metrics']['deep_learning'] = dl_performance
            results['diversity_metrics']['deep_learning'] = dl_diversity
        
        if xgb_trained:
            xgb_performance = self.evaluate_model_performance(xgboost_model, 'XGBoost')
            xgb_diversity = self.evaluate_recommendation_diversity(xgboost_model, 'XGBoost')
            
            results['performance_metrics']['xgboost'] = xgb_performance
            results['diversity_metrics']['xgboost'] = xgb_diversity
        
        return results
    
    def create_comparison_report(self, results: Dict) -> str:
        """
        Create a comprehensive comparison report
        
        Args:
            results: Results from run_comparison()
            
        Returns:
            String with formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("CAREER RECOMMENDATION SYSTEM - MODEL COMPARISON REPORT")
        report.append("=" * 80)
        
        # Training times
        report.append("\n1. TRAINING PERFORMANCE:")
        report.append("-" * 40)
        for model_name, training_time in results['training_times'].items():
            report.append(f"{model_name.replace('_', ' ').title()}: {training_time:.2f} seconds")
        
        # Performance metrics
        report.append("\n2. PREDICTION PERFORMANCE:")
        report.append("-" * 40)
        
        performance_data = []
        for model_name, metrics in results['performance_metrics'].items():
            if 'error' not in metrics:
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'MSE': f"{metrics['mse']:.4f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'R² Score': f"{metrics['r2_score']:.4f}",
                    'NDCG': f"{metrics['ndcg']:.4f}",
                    'MAP': f"{metrics['map_score']:.4f}",
                    'Avg Time (s)': f"{metrics['avg_prediction_time']:.4f}"
                })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            report.append(df.to_string(index=False))
        
        # Diversity metrics
        report.append("\n3. RECOMMENDATION DIVERSITY:")
        report.append("-" * 40)
        
        diversity_data = []
        for model_name, metrics in results['diversity_metrics'].items():
            diversity_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Coverage': f"{metrics['coverage']:.3f}",
                'Unique Courses': metrics['unique_courses_recommended'],
                'Avg Category Diversity': f"{metrics['avg_category_diversity']:.2f}"
            })
        
        if diversity_data:
            df = pd.DataFrame(diversity_data)
            report.append(df.to_string(index=False))
        
        # Summary and recommendations
        report.append("\n4. SUMMARY AND RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if len(results['performance_metrics']) >= 2:
            # Compare models
            dl_metrics = results['performance_metrics'].get('deep_learning', {})
            xgb_metrics = results['performance_metrics'].get('xgboost', {})
            
            if 'r2_score' in dl_metrics and 'r2_score' in xgb_metrics:
                if dl_metrics['r2_score'] > xgb_metrics['r2_score']:
                    report.append("• Deep Learning model shows better prediction accuracy (higher R² score)")
                else:
                    report.append("• XGBoost model shows better prediction accuracy (higher R² score)")
            
            if 'avg_prediction_time' in dl_metrics and 'avg_prediction_time' in xgb_metrics:
                if dl_metrics['avg_prediction_time'] < xgb_metrics['avg_prediction_time']:
                    report.append("• Deep Learning model is faster at making predictions")
                else:
                    report.append("• XGBoost model is faster at making predictions")
            
            # Training time comparison
            dl_training = results['training_times'].get('deep_learning', 0)
            xgb_training = results['training_times'].get('xgboost', 0)
            
            if dl_training > 0 and xgb_training > 0:
                if dl_training < xgb_training:
                    report.append("• Deep Learning model trains faster")
                else:
                    report.append("• XGBoost model trains faster")
        
        report.append("\n5. RECOMMENDATIONS FOR PRODUCTION:")
        report.append("-" * 40)
        report.append("• Consider ensemble approach combining both models")
        report.append("• Use A/B testing to validate real-world performance")
        report.append("• Monitor recommendation diversity and user satisfaction")
        report.append("• Implement continuous learning from user feedback")
        
        return "\n".join(report)

# Usage example
def main():
    # Load data
    students_df = pd.read_csv('student_data.csv')
    courses_df = pd.read_csv('Courses.csv')
    
    # Run comparison
    comparison = ModelComparison(students_df, courses_df)
    results = comparison.run_comparison()
    
    # Generate report
    report = comparison.create_comparison_report(results)
    print(report)
    
    # Save report to file
    with open('model_comparison_report.txt', 'w') as f:
        f.write(report)
    
    print("\nComparison complete! Report saved to 'model_comparison_report.txt'")

if __name__ == "__main__":
    main()
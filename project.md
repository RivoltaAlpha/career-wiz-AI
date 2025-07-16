# CareerWiz: AI-Powered Career Recommendation System for Kenyan High School Students

## Project Title
**CareerWiz: An AI-Based Career Recommendation Model for Kenyan High School Students**

## Project Overview
CareerWiz is an innovative AI-powered career recommendation system specifically designed for Kenyan high school students. The platform addresses the critical gap in structured career counseling within Kenyan educational institutions by providing personalized career guidance based on students' academic performance, interests, and the local job market dynamics.

### Problem Statement
- **Lack of structured career counseling** in Kenyan schools (less than 15% of 3+ million high school students access formal career guidance)
- **Mismatch between academic performance, personal interests, and career paths**
- **Career choices heavily influenced by peers or parents** rather than data-driven insights
- **Limited exposure to diverse career opportunities** that align with students' skills and interests
- **Absence of localized career guidance tools** tailored to the Kenyan curriculum and job market

### Solution
An AI-based career recommendation system that provides:
- Personalized career recommendations based on academic performance and interests
- Exposure to a wide range of career options aligned with the Kenyan job market
- Dynamic suggestions with continuous learning capabilities
- 24/7 accessible virtual career advisor

## Project Key Features

### Core Features
1. **User Authentication System**
   - Student registration and login
   - Secure profile management
   - Academic data storage and updates

2. **Academic Performance Analysis**
   - Subject-wise performance tracking
   - Grade analysis and trend identification
   - Curriculum-specific assessment (KCSE-aligned)

3. **Interest Assessment Module**
   - Interactive interest questionnaires
   - Personality trait analysis
   - Skill preference identification

4. **AI-Powered Recommendation Engine**
   - Machine learning-based career matching
   - Personalized course suggestions
   - University program recommendations

5. **Kenyan Job Market Integration**
   - Local industry insights
   - Salary expectations and growth prospects
   - Required qualifications and skills

6. **Feedback and Continuous Learning**
   - User feedback collection
   - Model performance improvement
   - Dynamic recommendation updates

### Advanced Features
- **Parent/Guardian Dashboard** (upcoming)
- **Mentorship Portal** (planned)
- **Career Path Visualization**
- **University Partnership Integration**
- **Mobile Application** (in development)

## Project Implementation Steps

### 1. Define the Problem and Scope
**Completed:**
- Problem identification and market research
- Target audience definition (Form 3 & 4 students, KCSE graduates)
- Scope limitation to Kenyan curriculum and job market

**Scope:**
- Primary: High school students (Form 3 & 4, KCSE graduates)
- Secondary: Educational institutions, parents, NGOs
- Geographic focus: Kenya with potential East African expansion

### 2. Data Collection ✅
**Available Datasets:**
- `student_data.csv`: Student academic and interest profiles
- `course_catalog.csv` / `Courses.csv`: Available courses and career paths
- Kenyan curriculum-specific subject mappings
- Local job market data integration

**Data Structure:**
- Student profiles: subjects, interests, academic performance
- Course catalog: course names, required subjects, skills, career outcomes
- Market data: industry trends, salary ranges, growth prospects

### 3. Model Selection

**Task:** Multi-class recommendation system for career path prediction

**Recommended Algorithms:**
1. **Content-Based Filtering**
   - K-Nearest Neighbors (KNN) - Currently implemented
   - Cosine Similarity for feature matching
   
2. **Collaborative Filtering**
   - Matrix Factorization for user-course interactions
   - Neural Collaborative Filtering for deep learning approach

3. **Hybrid Approaches**
   - Ensemble methods combining multiple algorithms
   - Deep learning models for complex pattern recognition

4. **Advanced ML Techniques**
   - Random Forest for feature importance analysis
   - Gradient Boosting for improved accuracy
   - Neural Networks for complex relationship modeling

**Recommended Libraries:**
- **Core ML:** `scikit-learn`, `pandas`, `numpy`
- **Deep Learning:** `TensorFlow`, `PyTorch`, `Keras`
- **NLP Processing:** `NLTK`, `spaCy` for text analysis
- **API Framework:** `FastAPI` (currently used), `Flask`
- **Database:** `PostgreSQL`, `SQLAlchemy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Cloud Deployment:** `Azure ML`, `Docker`, `Kubernetes`

### 4. Model Training and Evaluation

**Training Strategy:**
- **Data Split:** 80% training, 20% testing
- **Validation:** 5-fold cross-validation
- **Feature Engineering:** Subject-interest correlation analysis
- **Hyperparameter Tuning:** Grid search for optimal parameters

**Evaluation Metrics:**
- **Accuracy:** Overall prediction correctness
- **Precision:** Relevance of recommendations
- **Recall:** Coverage of suitable career options
- **F1-Score:** Balanced performance measure
- **NDCG:** Ranking quality assessment
- **User Satisfaction Score:** Feedback-based evaluation

**Cross-Validation:**
- K-fold cross-validation for model robustness
- Temporal validation for dynamic recommendations
- A/B testing for feature effectiveness

### 5. Build a User Interface

**Platform:** 
- **Frontend:** https://careerwiz-frontend.vercel.app/
- **Repository:** https://github.com/RivoltaAlpha/careerwiz-frontend
- **Backend:** https://github.com/RivoltaAlpha/Careerwiz-Backend

**Technology Stack:**
- **Frontend:** React.js, modern UI components
- **Backend:** FastAPI, Python
- **Database:** PostgreSQL
- **Authentication:** JWT tokens
- **Hosting:** Vercel (frontend), Azure (backend)

**User Interface Features:**
- Responsive design for multiple devices
- Intuitive navigation and user experience
- Interactive recommendation displays
- Progress tracking and analytics
- Multilingual support (English, Swahili)

### 6. Deployment

**Cloud Platform:** Microsoft Azure

**Deployment Architecture:**
- **App Service:** FastAPI backend hosting
- **Database:** Azure PostgreSQL
- **Storage:** Azure Blob Storage for static files
- **CDN:** Azure CDN for global content delivery
- **Monitoring:** Azure Application Insights
- **Security:** Azure Key Vault for secrets management

**CI/CD Pipeline:**
- GitHub Actions for automated deployment
- Docker containerization
- Automated testing and validation
- Blue-green deployment strategy

### 7. Testing and Validation

**Unit Testing:**
- Data preprocessing validation
- Model inference accuracy
- API endpoint functionality
- Database operations integrity

**Integration Testing:**
- End-to-end user flow testing
- API-frontend integration
- Database connectivity
- Third-party service integration

**User Testing:**
- Pilot program with 1 school (completed)
- Expansion to 10+ schools (planned)
- User feedback collection and analysis
- Performance optimization based on real usage

**Performance Testing:**
- Load testing for concurrent users
- Response time optimization
- Resource utilization monitoring
- Scalability assessment

### 8. Documentation and Reporting

**Documentation Requirements:**
- **Technical Documentation:** API documentation, code comments
- **User Manual:** Student and administrator guides
- **Deployment Guide:** Infrastructure setup and maintenance
- **Model Documentation:** Algorithm explanations and performance metrics

**Reporting Components:**
- **Project Methodology:** Detailed approach and implementation
- **Results Analysis:** Performance metrics and user feedback
- **Visualizations:** Charts, graphs, and system screenshots
- **Impact Assessment:** Educational and social impact measurement

**Presentation Materials:**
- **Stakeholder Presentations:** Ministry of Education, NGO partners
- **Technical Demos:** Live system demonstrations
- **Academic Presentations:** Research findings and methodologies
- **User Training Materials:** Workshops and training sessions

## Expected Outcomes

### 1. Functional AI-Powered System
- **Real-time career recommendations** based on student profiles
- **Scalable architecture** supporting thousands of concurrent users
- **High accuracy** in career-academic alignment predictions
- **Continuous learning** capabilities for improved recommendations

### 2. User-Friendly Application
- **Intuitive interface** accessible to high school students
- **Visual career path representations** with clear progression steps
- **Interactive dashboards** for progress tracking
- **Mobile-responsive design** for accessibility

### 3. Comprehensive Documentation
- **Detailed technical report** with methodology and results
- **User adoption metrics** and feedback analysis
- **Social impact assessment** aligned with UN SDGs
- **Scalability roadmap** for regional expansion

### 4. Social Impact Achievement
- **SDG 4 (Quality Education):** Improved educational decision-making
- **SDG 8 (Decent Work):** Better job market alignment
- **SDG 10 (Reduced Inequalities):** Equal access to career guidance
- **SDG 9 (Innovation):** Modernized educational guidance systems

## Current Status and Next Steps

### Completed Milestones
- ✅ Prototype development
- ✅ Initial model implementation (KNN-based)
- ✅ Basic API structure with FastAPI
- ✅ Frontend application deployment
- ✅ Pilot testing in 1 school

### Immediate Next Steps
1. **Model Enhancement:** Implement advanced algorithms and improve accuracy
2. **Data Expansion:** Gather more comprehensive training data
3. **Feature Development:** Add parent dashboards and mentorship portal
4. **Partnership Expansion:** Scale to 10+ schools
5. **Mobile Development:** Create native mobile applications
6. **Azure Deployment:** Migrate to production cloud infrastructure

### Long-term Goals
- **Regional Expansion:** Extend to other East African countries
- **AI Enhancement:** Implement deep learning models
- **Industry Integration:** Partner with employers for job placement
- **Government Adoption:** Integrate with Ministry of Education systems

## Success Metrics
- **User Adoption:** 10,000+ active student users within 6 months
- **Accuracy:** 85%+ recommendation accuracy rate
- **User Satisfaction:** 4.5/5 average rating
- **Educational Impact:** 20% improvement in course-career alignment
- **Market Penetration:** 50+ schools using the platform
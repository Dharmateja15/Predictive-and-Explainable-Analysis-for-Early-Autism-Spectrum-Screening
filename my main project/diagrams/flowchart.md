flowchart TD
    A[Start] --> B[Load dataset (CSV)]
    B --> C[Preprocess<br/>• Clean names<br/>• KNN impute<br/>• Label encode<br/>• SMOTE-Tomek]
    C --> D[Split Train/Test]
    D --> E[Train models<br/>NB / SVM / RF / ExtraTrees]
    E --> F[Cross-validate (ROC-AUC)]
    F --> G[Pick best model]
    G --> H[Evaluate on Test<br/>Report + ROC]
    H --> I[Save model + encoders + feature list]
    I --> J[User submits form in UI]
    J --> K[Preprocess to match training schema]
    K --> L{Predict}
    L -->|ASD| M[Show: ASD + prob]
    L -->|No ASD| N[Show: No ASD + prob]
    M --> O[Explain: SHAP + LIME]
    N --> O
    O --> P[Allow CSV/PNG export]
    P --> Q[End]

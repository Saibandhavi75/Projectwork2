## Title of the Project
Remaining Useful Life Estimation Using Ensemble Learning Approach for L-ion Batteries in Automobiles


The project focuses on predicting the Remaining Useful Life (RUL) of NMC-LCO Lithium-Ion batteries using machine learning models and ensemble learning techniques. Data from 14 batteries with a nominal capacity of 2.8 Ah is analyzed, extracting features like voltage decline rate and charge time change. K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) models are trained, achieving accuracies of 90.3%, 92.5%, and 88.7% respectively. The ensemble model ENHPT combines these predictions, resulting in a 94.8% accuracy. Deep learning models like CNN and LSTM further enhance accuracy to 96.2% and 97.5%. Feature selection through backward regression and correlation analysis optimizes model performance. Visualization tools like scatter plots and line graphs illustrate actual vs predicted RUL values, validating the models' effectiveness. This comprehensive approach supports advanced battery management systems for electric vehicles and renewable energy applications.
## About
<!--Detailed Description about the project-->
The precise evaluation of lithium-ion battery Remaining Useful Life (RUL) is crucial for optimizing performance and reliability in energy storage systems. This project focuses on predicting the RUL of NMC-LCO Lithium-Ion batteries using cycle data analysis. The dataset, sourced from the Hawaii Natural Energy Institute, contains information about 14 batteries with a nominal capacity of 2.8 Ah, subjected to CC-CV charging at C/2 discharge speeds under controlled conditions. Key features such as voltage decline rate, charge time change, and discharge duration are extracted to understand degradation patterns. Machine learning models including K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) are employed to predict RUL. Ensemble learning techniques like bagging, boosting, and stacking further enhance accuracy by combining predictions from individual models. The ENHPT framework achieves an accuracy of 94.8%, demonstrating the effectiveness of ensemble methods. Deep learning models like CNN and LSTM improve accuracy even further, achieving 96.2% and 97.5% respectively. This comprehensive approach supports advanced battery management systems for electric vehicles and renewable energy applications, enabling proactive maintenance and reducing operational costs. By integrating real-time sensor data and environmental factors, future research can refine these models for broader applicability and enhanced reliability.

## Features
<!--List the features of the project as shown below-->
1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Feature Scaling
5. Model Training (KNN, SVR, RF)
6. Model Evaluation (R², RMSE)
7.Visualization and Reporting



## Requirements
<!--List the requirements of the project as shown below-->
* Operating System : Requires a 64-bit OS (Windows 10 or Ubuntu) to ensure compatibility with machine learning frameworks and libraries.
* Development Environment : Python 3.7 or later is necessary for implementing the battery RUL prediction system.
* Machine Learning Libraries : Scikit-learn for traditional machine learning models like KNN, Random Forest, and SVM; TensorFlow for deep learning models such as CNN and LSTM.
* Data Processing Libraries : Pandas and NumPy for data manipulation and analysis.
* Visualization Libraries : Matplotlib and Seaborn for creating visualizations of model performance and feature importance.
* Version Control : Implementation of Git for collaborative development and effective code management.
* IDE : Use of Jupyter Notebook or VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies : Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Statsmodels for various tasks in data processing, visualization, and statistical analysis.

## System Architecture
<!--Embed the system architecture diagram as shown below-->
![image](https://github.com/user-attachments/assets/f0a1dfd3-aa6d-43a4-bc66-2d65717e3650)


## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Correlation Heat map between parameters in estimation of RUL

![image](https://github.com/user-attachments/assets/24ffe5ff-b056-4409-b118-ad17261ad170)


#### Output2 - Maximum Voltage During Discharge vs RUL
![image](https://github.com/user-attachments/assets/fe3049f0-3a6f-4d77-b81d-616fcb75bb84)


#### Output3 - Minimum Voltage During Discharge vs RUL
![image](https://github.com/user-attachments/assets/3f0679ac-dc70-483b-8a09-ae293db19208)



## Results and Impact
<!--Give the results and impact as shown below-->
The project successfully evaluated the Remaining Useful Life (RUL) of NMC-LCO Lithium-Ion batteries using various machine learning models, achieving accuracies up to 97.5% with LSTM. Ensemble methods further enhanced prediction reliability, reaching 94.8% accuracy. Detailed analysis through scatter plots and line graphs validated model performance, showing strong alignment between actual and predicted RUL values. Feature importance analysis highlighted critical parameters affecting battery degradation, such as maximum discharge voltage and minimum charging voltage.

This research significantly advances battery management systems by providing accurate RUL predictions, enabling proactive maintenance and reducing unexpected failures in electric vehicles and renewable energy storage. Enhanced predictive capabilities lead to cost savings and improved operational efficiency, supporting sustainable energy solutions. By incorporating real-time sensor data and environmental factors, future applications can achieve even higher accuracy and reliability.

## Articles published / References
[1] Safavi, V., Mohammadi Vaniar, A., Bazmohammadi, N., Vasquez, J. C., & Guerrero, J. M. (2024). Battery remaining useful life prediction using machine learning models: A comparative study. Information, 15(3), 124. https://doi.org/10.3390/info15030124
[2]Zhang, Y., & Li, H. (2021). AI-based control of environmental parameters in submarine cabins: A review. Renewable and Sustainable Energy Reviews, 135, 110195.
[3]Suh, S., Mittal, D. A., Bello, H., Zhou, B., Jha, M. S., & Lukowicz, P. (2023). Remaining useful life prediction of lithium-ion batteries using spatio-temporal multimodal attention networks. arXiv preprint arXiv:2310.18924.
[4]Mittal, D., Bello, H., Zhou, B., Jha, M. S., Suh, S., & Lukowicz, P. (2023). Two-stage early prediction framework of remaining useful life for lithium-ion batteries. arXiv preprint arXiv:2308.03664.
[5]Hilal, H., & Saha, P. (2023). Forecasting lithium-ion battery longevity with limited data availability: Benchmarking different machine learning algorithms. arXiv preprint arXiv:2312.05717.



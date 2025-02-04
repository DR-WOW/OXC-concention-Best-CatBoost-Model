import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model_path = "cat_grid_search.pkl"
model = joblib.load(model_path)

# 获取最佳估计器
best_model = model.best_estimator_

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Concentration Prediction", page_icon="📊")
st.title("📊 Concentration Prediction and SHAP Visualization")
st.write("""
Through inputting feature values, you can get the model's prediction and understand the contribution of each feature using SHAP analysis.
""")

# 特征输入区域
st.sidebar.header("Feature Input Area")
st.sidebar.write("Please input feature values:")

# 定义特征输入范围
feature_ranges = {
    "SEX": {"type": "categorical", "options": [0, 1], "default": 0, "description": "Gender (0 = Female, 1 = Male)"},
    "AGE": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0, "description": "Age of the patient (in years)"},
    "WT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 25.0, "description": "Weight of the patient (in kg)"},
    "Single_Dose": {"type": "numerical", "min": 0.0, "max": 60.0, "default": 15.0, "description": "Single dose of the drug per weight (mg/kg)"},
    "Daily_Dose": {"type": "numerical", "min": 0.0, "max": 2400.0, "default": 450.0, "description": "Total daily dose of the drug (mg)"},
    "SCR": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 30.0, "description": "Serum creatinine level (mg/dL)"},
    "CLCR": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 90.0, "description": "Creatinine clearance rate (mL/min)"},
    "BUN": {"type": "numerical", "min": 0.0, "max": 50.0, "default": 5.0, "description": "Blood urea nitrogen level (mg/dL)"},
    "ALT": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "Alanine aminotransferase level (U/L)"},
    "AST": {"type": "numerical", "min": 0.0, "max": 150.0, "default": 18.0, "description": "Aspartate transaminase level (U/L)"},
    "CL": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 3.85, "description": "Metabolic clearance rate of the drug (mL/min)"},
    "V": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 10.0, "description": "Apparent volume of distribution of the drug (L)"}
}

# 动态生成输入界面
inputs = {}
for feature, config in feature_ranges.items():
    if config["type"] == "numerical":
        inputs[feature] = st.sidebar.number_input(
            f"{feature} ({config['description']})",
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"]
        )
    elif config["type"] == "categorical":
        inputs[feature] = st.sidebar.selectbox(
            f"{feature} ({config['description']})",
            options=config["options"],
            index=config["options"].index(config["default"])
        )

# 将输入特征转换为 Pandas DataFrame
features_df = pd.DataFrame([inputs])

# 如果模型在训练时使用了分类特征，确保这些特征是整数类型
cat_features = ["SEX"]  # 假设 SEX 是分类特征
features_df[cat_features] = features_df[cat_features].astype(int)

# 模型预测
if st.button("Predict"):
    try:
        prediction = best_model.predict(features_df)[0]  # 预测结果是连续性变量

        # 显示预测结果
        st.header("Prediction Result")
        st.success(f"Based on the feature values, the predicted concentration is {prediction:.2f} mg/L.")

        # 保存预测结果为图像
        fig, ax = plt.subplots(figsize=(8, 1))
        text = f"Predicted Concentration: {prediction:.2f} mg/L"
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes
        )
        ax.axis('off')
        plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
        st.image("prediction_text.png")

        # 计算 SHAP 值
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(features_df)

            # 生成 SHAP 力图
            html_output = shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                features_df.iloc[0, :],
                show=False
            )

            # 将 HTML 输出保存为文件
            with open("shap_force_plot.html", "w") as f:
                f.write(html_output.html())

            # 在 Streamlit 中显示 HTML 文件
            st.components.v1.html(html_output.html(), height=400)

        except Exception as e:
            st.error(f"An error occurred during SHAP visualization: {e}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

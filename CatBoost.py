import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib


# 加载模型
model_path = "cat_grid_search.pkl"
CatBoost = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="CatBoost 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 CatBoost 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
SEX = st.sidebar.selectbox("性别 Gender(1 = male, 0 = female)", [0, 1])
AGE= st.sidebar.number_input("年龄Age (范围: 0.0-18)", min_value=0.0, max_value=18.0, value=5.0)
WT = st.sidebar.number_input("体重Weight (范围: 0.0-100.0)", min_value=0.0, max_value=100.0, value=25.0)
Single_Dose = st.sidebar.number_input("单次给药剂量/体重Single_Dose/weight (范围: 0.0-60)", min_value=0.0, max_value=60, value=15.0)
Daily_Dose = st.sidebar.number_input("日总剂量Daily_Dose (范围: 0.0-2400)", min_value=0.0, max_value=2400, value=450)
SCR = st.sidebar.number_input("血清肌酐Serum creatinine (范围: 0.0-150.00)", min_value=0.0, max_value=150.0, value=30.0)
CLCR = st.sidebar.number_input("肌酐清除率Creatinine clearance rate (范围: 0.0-200.00)", min_value=0.0, max_value=200.00, value=90.00)
BUN = st.sidebar.number_input("血尿素氮 (范围: 0.0-50.0)", min_value=0.0, max_value=50.0, value=5.0)
ALT = st.sidebar.number_input("丙氨酸氨基转移酶Alanine aminotransferase (ALT) (范围: 0.0-150.0)", min_value=0.0, max_value=150.0, value=18.0)
AST = st.sidebar.number_input("天冬氨酸氨基转移酶Aspartate transaminase (AST) (范围: 0.0-150.0)", min_value=0.0, max_value=150.0, value=18.0)
CL = st.sidebar.number_input("药物的代谢清除率 Metabolic clearance of drugs (CL)(范围: 0.0-20.0)", min_value=0.0, max_value=100.0, value=3.85)
V = st.sidebar.number_input("药物的表观分布容积(Vd)(范围: 0.0-1000.0)", min_value=0.0, max_value=1000.0, value=10.0)

# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("浓度预测结果(mg/L)")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([SEX, AGE, WT, Single_Dose,	Daily_Dose, SCR, CLCR,	BUN	, ALT, AST, CL, V]).reshape(1, -1)


        # 模型预测
        prediction = CatBoost.predict(input_array)[0]

        # 显示预测结果
        st.success(f"预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# 可视化展示
st.header("SHAP 可视化与模型预测性能分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括各个特征贡献 SHAP 可视化
、模型预测的相对准确度以及模型预测的绝对准确度。
""")

# 各个特征影响重要性 SHAP 可视化
st.subheader("1. 各个特征影响重要性")
st.write("各个特征贡献与影响程度分析。")
first_layer_img = "SHAP combined with top line corrected of Catboost Model.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="各个特征的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到各个特征的 SHAP 贡献分析图像文件。")

# 相对准确度可视化
st.subheader("2.模型预测的相对准确度可视化")
st.write("模型预测的相对准确度（±30%）。")
Relative_Accuracy_img = "30% Relative Accuracy Analysis of Catboost Model.png"
try:
    img2 = Image.open(Relative_Accuracy_img)
    st.image(img2, caption="模型预测的相对准确度（±30%）分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到模型预测的相对准确度（±30%）图像文件。")

# 绝对准确度可视化
st.subheader("3. 模型预测的绝对准确度可视化")
st.write("模型预测的绝对准确度（±3 mg/mL）。")
Absolute_accuracy_img = " Absolute accuracy （±3 mg/mL）Analysis of Catboost model.png"
try:
    img3 = Image.open(Absolute_accuracy_img)
    st.image(img3, caption="模型预测的绝对准确度（±3 mg/mL）分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到模型预测的绝对准确度（±3 mg/mL）图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解模型的 SHAP 分析结果，包括各个特征贡献 SHAP 可视化
、模型预测的相对准确度以及模型预测的绝对准确度等情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")

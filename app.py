import sys
import io
# উইন্ডোজ টার্মিনালে ইউনিকোড এনকোডিং ঠিক রাখার জন্য
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# এন্টারপ্রাইজ মডেল প্যাকেজ ফাইল লোড করা
try:
    with open('summit_enterprise_model.pkl', 'rb') as f:
        enterprise_package = pickle.load(f)
    
    preprocessor = enterprise_package['preprocessing_pipeline']
    le_weather = enterprise_package['label_encoder_weather']
    model_congest = enterprise_package['model_congest']
    model_speed = enterprise_package['model_speed']
    model_fault = enterprise_package['model_fault']
    column_order = enterprise_package['column_order']
    
    print("[SUCCESS] Summit Enterprise Model Package loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model package: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    
    tower_id = int(req['tower_id'])
    users_connected = float(req['users_connected'])
    current_download = float(req['download_speed'])
    current_upload = float(req['upload_speed'])
    latency = float(req['latency'])
    weather_str = req['weather']
    target_hour = int(req['target_hour'])
    
    # আবহাওয়া এনকোডিং হ্যান্ডলিং
    try:
        encoded_weather = le_weather.transform([weather_str])[0]
    except:
        encoded_weather = 0
        
    # পিক আওয়ার ক্যালকুলেশন (সন্ধ্যা ৬টা থেকে রাত ১১টা)
    is_peak = 1 if (18 <= target_hour <= 23) else 0
    
    # ফল্ট রিস্ক এবং ইউজার ট্রাফিক অনুপাত হিসাব
    fault_risk_pct = min(max(latency * 0.2 + (1.0 if weather_str in ['Rain', 'Storm', 'Snow'] else 0.0) * 15.0, 0), 100)
    user_traffic_ratio = current_download / (users_connected + 1e-5)
    
    input_df = pd.DataFrame([{
        'tower_id': tower_id,
        'users_connected': users_connected,
        'weather': encoded_weather,
        'download_speed': current_download,
        'upload_speed': current_upload,
        'latency': latency,
        'fault_risk_pct': fault_risk_pct,
        'user_traffic_ratio': user_traffic_ratio,
        'is_peak_hour': is_peak
    }])
    
    # কলাম সিকোয়েন্স মেলানো
    input_df = input_df[column_order]
    
    # ইনপুটের ওপর ভিত্তি করে ডায়নামিক স্পিড ও কনজেশন ক্যালকুলেশন
    user_load_factor = users_connected / 500.0  
    peak_multiplier = 1.3 if is_peak else 1.0
    weather_impact = 1.2 if weather_str in ['Rain', 'Storm', 'Snow'] else 1.0
    
    # প্রেডিকটেড স্পিড হিসাব
    predicted_speed = max(5.0, current_download - (user_load_factor * 15.0 * peak_multiplier * weather_impact))
    predicted_speed = round(min(predicted_speed, current_download), 2)
    
    # স্পিড ড্রপ পার্সেন্টেজ
    speed_drop_pct = max(0.0, round(((current_download - predicted_speed) / current_download) * 100, 2))
    
    # কনজেশন পার্সেন্টেজ (ইউজার এবং ল্যাটেন্সির ওপর নির্ভর করে পরিবর্তিত হবে)
    congestion_calculated = (users_connected / 999.0) * 60.0 + (latency / 200.0) * 25.0 + (15.0 if is_peak else 0.0)
    congestion_pct = round(min(max(congestion_calculated, 5.0), 98.0), 2)
    
    # ফল্ট রিস্ক প্রবাবিলিটি
    fault_calculated = (latency / 200.0) * 55.0 + (35.0 if weather_str in ['Rain', 'Storm', 'Snow'] else 5.0)
    fault_risk_pred = round(min(max(fault_calculated, 2.0), 95.0), 2)
    
    # প্রয়োজনীয় ব্যান্ডউইথ (Mbps) হিসাব
    required_mbps_for_no_jam = round(max(current_download, users_connected * 0.15) * peak_multiplier, 2)
    
    # ৩-লাইন বাইপাস লজিক
    bypass_needed = False
    if congestion_pct > 70.0 or speed_drop_pct > 30.0 or fault_risk_pred > 75.0:
        bypass_needed = True
        
    return jsonify({
        "target_hour": target_hour,
        "speed_drop_pct": speed_drop_pct,
        "congestion_pct": congestion_pct,
        "predicted_speed": predicted_speed,
        "required_mbps": required_mbps_for_no_jam,
        "fault_risk_pct": fault_risk_pred,
        "bypass": bypass_needed
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
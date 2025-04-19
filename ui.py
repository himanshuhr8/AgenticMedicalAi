import streamlit as st
import json
from agentclinic import ui_main  # Assuming this is the location of the ui_main function.

st.set_page_config(page_title="Agentic AI - Medical OSCE Diagnosis", layout="wide")

st.title("🧠 Agentic Medical AI — OSCE Case Diagnosis")
st.markdown("Demonstration of a multi-agent AI reasoning system on structured patient scenarios.")

# Sample pre-filled scenario
sample_case = {
    "OSCE_Examination": {
        "Objective_for_Doctor": "Assess and diagnose the patient presenting with double vision, difficulty climbing stairs, and upper limb weakness.",
        "Patient_Actor": {
            "Demographics": "35-year-old female",
            "History": "1-month history of double vision, difficulty climbing stairs, and weakness when brushing her hair. Symptoms worsen after activity, improve with rest.",
            "Symptoms": {
                "Primary_Symptom": "Double vision",
                "Secondary_Symptoms": [
                    "Difficulty climbing stairs",
                    "Weakness in upper limbs",
                    "Improvement of symptoms after rest"
                ]
            },
            "Past_Medical_History": "No significant past medical history.",
            "Social_History": "Non-smoker, drinks wine occasionally. Works as a graphic designer.",
            "Review_of_Systems": "No chest pain, palpitations, shortness of breath, or infections."
        },
        "Physical_Examination_Findings": {
            "Vital_Signs": {
                "Temperature": "36.6°C",
                "Blood_Pressure": "125/80 mmHg",
                "Heart_Rate": "72 bpm",
                "Respiratory_Rate": "16 breaths/min"
            },
            "Neurological_Examination": {
                "Cranial_Nerves": "Ptosis of right upper eyelid, worsens with upward gaze.",
                "Motor_Strength": "Weakness in upper limbs, no atrophy.",
                "Reflexes": "Normal",
                "Sensation": "Normal"
            }
        },
        "Test_Results": {
            "Blood_Tests": {
                "Acetylcholine_Receptor_Antibodies": "Present (elevated)"
            },
            "Electromyography": {
                "Findings": "Decreased muscle response with repetitive stimulation"
            },
            "Imaging": {
                "Chest_CT": {
                    "Findings": "Normal, no thymoma"
                }
            }
        },
        "Correct_Diagnosis": "Myasthenia gravis"
    }
}

# Sidebar: scenario input
st.sidebar.header("📋 Scenario Input")
use_sample = st.sidebar.checkbox("Use Sample Case", value=True)
uploaded_json = st.sidebar.file_uploader("Upload JSON File", type=["json"])

scenario = None

if use_sample:
    scenario = sample_case
elif uploaded_json is not None:
    scenario = json.load(uploaded_json)

if scenario:
    # Display scenario summary
    osce = scenario["OSCE_Examination"]
    st.subheader("🧑‍⚕️ Doctor's Objective")
    st.info(osce["Objective_for_Doctor"])

    st.subheader("🧍‍♀️ Patient Details")
    patient = osce["Patient_Actor"]
    st.markdown(f"- **Demographics:** {patient['Demographics']}")
    st.markdown(f"- **Primary Symptom:** {patient['Symptoms']['Primary_Symptom']}")
    st.markdown(f"- **Other Symptoms:** {', '.join(patient['Symptoms']['Secondary_Symptoms'])}")
    with st.expander("Full Patient History"):
        st.markdown(f"**History:** {patient['History']}")
        st.markdown(f"**Past Medical History:** {patient['Past_Medical_History']}")
        st.markdown(f"**Social History:** {patient['Social_History']}")
        st.markdown(f"**Review of Systems:** {patient['Review_of_Systems']}")

    st.subheader("🧪 Physical & Test Findings")
    with st.expander("Physical Examination"):
        st.json(osce["Physical_Examination_Findings"])
    with st.expander("Test Results"):
        st.json(osce["Test_Results"])
gemini_api_key = st.secrets["GEMINI_API_KEY"]

if st.button("🤖 Run Agent Diagnosis"):
    with st.spinner("Agent is thinking..."):
        try:
            # Configuration parameters
            gemini_api_key = gemini_api_key
            doctor_llm = "gemini-2.0-flash"
            patient_llm = "gemini-2.0-flash"
            measurement_llm = "gemini-2.0-flash"
            moderator_llm = "gemini-2.0-flash"
            inf_type = "agent"
            doctor_bias = False
            patient_bias = False
            img_request = False
            total_inferences = 5
            replicate_api_key = None
            openai_api_key = None
            anthropic_api_key = None
            num_scenarios = 1

            # Run the agent-based inference system
            result = ui_main(
                scenario_dict=scenario,
                gemini_api_key=gemini_api_key,
                api_key=openai_api_key,
                replicate_api_key=replicate_api_key,
                inf_type=inf_type,
                doctor_bias=doctor_bias,
                patient_bias=patient_bias,
                doctor_llm=doctor_llm,
                patient_llm=patient_llm,
                measurement_llm=measurement_llm,
                moderator_llm=moderator_llm,
                num_scenarios=num_scenarios,
                img_request=img_request,
                total_inferences=total_inferences,
                anthropic_api_key=anthropic_api_key
            )

            # ✅ Show final diagnosis
            # ✅ Show final diagnosis
            st.success(f"✅ Diagnosis Ready: **{result['Diagnosis']}**")
            st.markdown("> This diagnosis was generated based on patient symptoms, physical exam, and test findings using agentic step-wise inference.")

            # 💬 Show doctor-patient chat conversation
            # 💬 Show doctor-patient chat conversation (Enhanced UI)
            st.subheader("💬 Conversation")

            # Custom CSS for left-right chat alignment
            st.markdown("""
            <style>
            .chat-container {
                display: flex;
                margin-bottom: 1rem;
            }
            .chat-left {
                justify-content: flex-start;
            }
            .chat-right {
                justify-content: flex-end;
            }
            .bubble {
                padding: 0.7rem 1rem;
                border-radius: 1rem;
                max-width: 65%;
                font-size: 0.95rem;
                line-height: 1.4;
                box-shadow: 0px 1px 4px rgba(0,0,0,0.15);
                color: black; /* <-- Ensure black text */
            }
            .doctor {
                background-color: #e0f2ff;
                border-top-left-radius: 0;
            }
            .patient {
                background-color: #e6f9e9;
                border-top-right-radius: 0;
            }
            .other {
                background-color: #f2f2f2;
                border-top-left-radius: 0;
            }
            .speaker-label {
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 0.3rem;
            }
            </style>
            """, unsafe_allow_html=True)


            # Loop through steps and align based on role
            for step in result.get("Reasoning", []):
                role = step["step"].lower()
                text = step["details"]

                if "doctor" in role:
                    alignment = "chat-left"
                    role_class = "doctor"
                    label = "🧑‍⚕️ Doctor"
                elif "patient" in role:
                    alignment = "chat-right"
                    role_class = "patient"
                    label = "🧍 Patient"
                elif "measurement" in role:
                    alignment = "chat-left"
                    role_class = "other"
                    label = "📊 Measurement"
                elif "moderator" in role:
                    alignment = "chat-left"
                    role_class = "other"
                    label = "🧑‍💼 Moderator"
                else:
                    alignment = "chat-left"
                    role_class = "other"
                    label = "🤖 System"

                st.markdown(f"""
                <div class="chat-container {alignment}">
                    <div class="bubble {role_class}">
                        <div class="speaker-label">{label}</div>
                        {text}
                    </div>
                </div>
                """, unsafe_allow_html=True)


            # 💡 Show reasoning steps again (optional)
            with st.expander("🧠 Diagnostic Reasoning Steps (Raw Logs)"):
                for step in result["Reasoning"]:
                    st.markdown(f"- **{step['step']}**: {step['details']}")


        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

else:
    st.warning("Please select 'Use Sample Case' or upload a scenario JSON file.")

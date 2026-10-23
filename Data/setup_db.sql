
    DROP TABLE IF EXISTS Clinical_Measurements_Temporal CASCADE;
    DROP TABLE IF EXISTS Visits CASCADE;
    DROP TABLE IF EXISTS Patients CASCADE;
    DROP TABLE IF EXISTS extracted_radiology_features CASCADE; 

    CREATE TABLE Patients (
        patient_id INT PRIMARY KEY,
        age INT,
        diabetes_mellitus INT,
        immunosuppression INT,
        cirrhosis INT,
        hepatic_failure INT,
        leukemia INT,
        lymphoma INT,
        solid_tumor_with_metastasis INT
    );

    CREATE TABLE Visits (
        hadm_id INT PRIMARY KEY,
        patient_id INT REFERENCES Patients(patient_id),
        hospital_outcome_death INT
    );

    CREATE TABLE Clinical_Measurements_Temporal (
        id SERIAL PRIMARY KEY,
        patient_id INT,
        hadm_id INT REFERENCES Visits(hadm_id),
        gcs_eyes INT,
        gcs_motor INT,
        gcs_verbal INT,
        sysbp_min FLOAT,
        heart_rate_max FLOAT,
        spo2_min FLOAT,
        glucose_max FLOAT,
        glucose_min FLOAT,
        d1_glucose_max FLOAT,
        d1_glucose_min FLOAT,
        d1_sysbp_max FLOAT,
        d1_heartrate_min FLOAT,
        d1_diasbp_min FLOAT,
        age INT,
        is_intubated INT,
        ventilated_apache INT,
        apache_3j_bodysystem VARCHAR(50),
        valid_to TIMESTAMP DEFAULT '2099-12-31 00:00:00'
    );

    -- Tabela cech radiologicznych i NLP z uwzględnieniem nerwu okoruchowego
    CREATE TABLE IF NOT EXISTS extracted_radiology_features (
        hadm_id INT PRIMARY KEY,
        vision_midline_shift_mm FLOAT,
        vision_basal_cisterns FLOAT,
        vision_corpus_callosum FLOAT,
        nlp_c_spine_fracture FLOAT,
        nlp_brain_aneurysm FLOAT,
        nlp_oculomotor_palsy FLOAT
    ); 
CREATE TABLE IF NOT EXISTS employee (
    e_id          SERIAL PRIMARY KEY,
    e_name        VARCHAR(100) NOT NULL,
    e_password    VARCHAR(100) NOT NULL,
    e_gender      VARCHAR(10),
    e_salary_ph   NUMERIC(10,2) NOT NULL DEFAULT 0,   
    e_attendance  VARCHAR(10)  NOT NULL DEFAULT 'Absent'  
);

CREATE TABLE IF NOT EXISTS attendance (
    a_id        SERIAL PRIMARY KEY,
    e_id        INTEGER NOT NULL REFERENCES employee(e_id),
    attendance  VARCHAR(10) NOT NULL,      
    start_time  TIMESTAMP NOT NULL,
    end_time    TIMESTAMP                  
);

CREATE TABLE IF NOT EXISTS salary (
    s_id                SERIAL PRIMARY KEY,
    e_id                INTEGER NOT NULL REFERENCES employee(e_id),
    e_salary_ph         NUMERIC(10,2),
    office_hours        NUMERIC(6,2),
    overtime_hours      NUMERIC(6,2),
    office_ph_salary    NUMERIC(10,2),
    overtime_ph_salary  NUMERIC(10,2),
    total_salary        NUMERIC(10,2),
    record_date         DATE DEFAULT CURRENT_DATE
);


INSERT INTO employee (e_name, e_password, e_gender, e_salary_ph)
VALUES ('User 1', '1234', 'M', 20.00)
ON CONFLICT DO NOTHING;

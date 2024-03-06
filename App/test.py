import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(
    spreadsheet=st.secrets["spreadsheet"],
)

# Print results.
print(df.columns)
for index, row in df.iterrows():
    st.write(f"{row['Emp_id']} has a :{row['Vehicle_Number']}:")
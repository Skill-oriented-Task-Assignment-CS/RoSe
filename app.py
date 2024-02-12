from flask import Flask, render_template, request, send_file, make_response
import pandas as pd
import os
import io
from deployment.SynthesisLib import *
import zipfile


app = Flask(__name__)


def generate_single_file_response(df) :
     #Convert DataFrame to CSV string
    csv_data = df.to_csv(index=False)
    # Create a response with the CSV data
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    response.headers['Content-type'] = 'text/csv'
    return response 

def generate_zip_file_response(df1,df2) :
     #Convert DataFrame to CSV string
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('data_task.csv', df1.to_csv(index=False))
        zip_file.writestr('data_worker.csv', df2.to_csv(index=False))

    # Set the buffer's position to the start
    zip_buffer.seek(0)
    return send_file(zip_buffer, download_name='data.zip', as_attachment=True)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form input values
        # radio_value = request.form.get('radio')
        checkbox_values = request.form.getlist('checkbox')
        input_text_value = request.form.get('input_text')
        # 0 - worksers
        # 1 - jobs

        if 'checkbox1' in checkbox_values:
            index=0
            df_worker = generate(index,int(input_text_value))
            synfile=generate_single_file_response(df_worker)

        elif 'checkbox2' in checkbox_values:
            index=1
            df_task = generate(index,int(input_text_value))
            synfile=generate_single_file_response(df_task)

        elif 'checkbox3' in checkbox_values:
            df_task = generate(0,int(input_text_value))
            df_worker = generate(1,int(input_text_value))
            synfile=generate_zip_file_response(df_task,df_worker)

        else :
            return "Server Error"
        
        return synfile
       

    return render_template('index.html')



def convert_to_csv(fake_data):

   # Convert the DataFrame to CSV format
    csv_data = fake_data.to_csv(index=False)
    csv_stream = io.BytesIO()
    csv_stream.write(csv_data.encode())
    csv_stream.seek(0)
    return csv_stream

if __name__ == '__main__':
    # app.run(port=5000,host="0.0.0.0")
   
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


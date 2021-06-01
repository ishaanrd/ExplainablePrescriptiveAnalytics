FROM python:3

COPY requirements.txt /

RUN pip install -r requirements.txt

ADD SJM_Anonymizer_2602ID.py /

ADD sampleinput.csv /

CMD ["python","./SJM_Anonymizer_2602ID.py","./sampleinput.csv"]
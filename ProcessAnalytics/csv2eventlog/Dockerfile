FROM python:3 

COPY requirements.txt /

RUN pip install -r requirements.txt 

ADD CSV2EL_Parser.py /

ADD 20210309-122708-sjm-anonymized.csv /

ENTRYPOINT ["python","./CSV2EL_Parser.py","./20210309-122708-sjm-anonymized.csv"]
 
CMD ["generic"]
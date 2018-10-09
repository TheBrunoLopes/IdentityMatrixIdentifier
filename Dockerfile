FROM python:3.6.6-slim-jessie
WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "IdentityMatrixIdentifier/main.py" ]
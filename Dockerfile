FROM mcr.microsoft.com/dotnet/framework/runtime:4.8

ADD https://aka.ms/vs/16/release/vc_redist.x64.exe /vc_redist.x64.exe

RUN C:\vc_redist.x64.exe /quiet /install

RUN setx path "%path%;C:\Windows\System32"

FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=flask_server.py

CMD flask run --host=0.0.0.0


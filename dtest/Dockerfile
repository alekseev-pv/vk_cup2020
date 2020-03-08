FROM continuumio/anaconda3:latest

# disable to avoid 300Mb uploads
#RUN pip install xgboost
#RUN pip install lightgbm
#RUN pip install catboost
#RUN pip install pypy

CMD mkdir /opt/results
WORKDIR /opt/results
COPY baseline_always_zero.py baseline_always_zero.py
COPY r1.pkl r1.pkl
COPY r2.pkl r2.pkl
COPY r3.pkl r3.pkl

COPY v1.pkl v1.pkl
COPY v2.pkl v2.pkl

COPY usersT.npy usersT.npy

COPY usersH.npy usersH.npy
COPY usersH1.npy usersH1.npy

COPY usersCPMmin.npy usersCPMmin.npy
COPY usersCPMmax.npy usersCPMmax.npy

COPY usersHours24.npy usersHours24.npy
COPY usersCPM24.npy usersCPM24.npy

COPY usersViews24.npy usersViews24.npy

CMD python main.py /tmp/data/test.tsv > /opt/results/result.tsv

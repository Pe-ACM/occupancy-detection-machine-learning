import os, time, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(BASE,'resultados_occupancy')
os.makedirs(OUT,exist_ok=True)

def load(name):
    df=pd.read_csv(os.path.join(BASE,name), index_col=0)
    df.columns=[c.strip() for c in df.columns]
    df['date']=pd.to_datetime(df['date'])
    return df
train=load('datatraining.txt')
test1=load('datatest.txt')
test2=load('datatest2.txt')
test=pd.concat([test1,test2],ignore_index=True)
features=['Temperature','Humidity','Light','CO2','HumidityRatio']
Xtr,ytr=train[features],train['Occupancy']
Xte,yte=test[features],test['Occupancy']

models={
 'Árvore de Decisão': DecisionTreeClassifier(max_depth=6,min_samples_leaf=5,random_state=42),
 'Random Forest': RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=-1),
 'KNN': Pipeline([('scale',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=7,weights='distance'))])
}
rows=[]; reports={}; cms={}
for name,model in models.items():
    t0=time.perf_counter(); model.fit(Xtr,ytr); train_time=time.perf_counter()-t0
    t0=time.perf_counter(); pred=model.predict(Xte); pred_time=time.perf_counter()-t0
    rows.append({'Algoritmo':name,'Acurácia':accuracy_score(yte,pred),'Precisão':precision_score(yte,pred,average='weighted',zero_division=0),'Recall':recall_score(yte,pred,average='weighted',zero_division=0),'F1-score':f1_score(yte,pred,average='weighted',zero_division=0),'Tempo de treino (s)':train_time,'Tempo de predição (s)':pred_time})
    reports[name]=classification_report(yte,pred,output_dict=True,zero_division=0)
    cms[name]=confusion_matrix(yte,pred).tolist()
res=pd.DataFrame(rows).sort_values('F1-score',ascending=False)
res.to_csv(os.path.join(OUT,'metricas_modelos.csv'),index=False)
with open(os.path.join(OUT,'relatorios_por_classe.json'),'w',encoding='utf-8') as f: json.dump(reports,f,ensure_ascii=False,indent=2)
with open(os.path.join(OUT,'matrizes_confusao.json'),'w',encoding='utf-8') as f: json.dump(cms,f,ensure_ascii=False,indent=2)

# dataset summary
all_df=pd.concat([train,test1,test2],ignore_index=True)
summary=[]
for c in all_df.columns:
    s=all_df[c]
    if c=='date': vals=f"{s.min()} a {s.max()}"; typ='data/hora'
    elif c=='Occupancy': vals='0 (não ocupado), 1 (ocupado)'; typ='binário'
    else: vals=f"{s.min():.6g} a {s.max():.6g}"; typ='numérico contínuo'
    summary.append({'Atributo':c,'Tipo':typ,'Intervalo/valores':vals,'Faltantes':int(s.isna().sum())})
pd.DataFrame(summary).to_csv(os.path.join(OUT,'caracterizacao_atributos.csv'),index=False)

# plots
plt.figure(figsize=(6,4)); all_df['Occupancy'].value_counts().sort_index().plot(kind='bar',color=['#58728c','#d08b3e']); plt.title('Distribuição do atributo classe'); plt.xlabel('Occupancy'); plt.ylabel('Quantidade'); plt.xticks([0,1],['0 - Não ocupado','1 - Ocupado'],rotation=0); plt.tight_layout(); plt.savefig(os.path.join(OUT,'distribuicao_classe.png'),dpi=180); plt.close()

corr=all_df[features+['Occupancy']].corr()
plt.figure(figsize=(7,5)); im=plt.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(corr)),corr.columns,rotation=45,ha='right'); plt.yticks(range(len(corr)),corr.index); 
for i in range(len(corr)):
 for j in range(len(corr)): plt.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
plt.title('Matriz de correlação'); plt.tight_layout(); plt.savefig(os.path.join(OUT,'matriz_correlacao.png'),dpi=180); plt.close()

plot=res.set_index('Algoritmo')[['Acurácia','Precisão','Recall','F1-score']]
ax=plot.plot(kind='bar',figsize=(8,4.8),ylim=(0.85,1.005),color=['#2f6690','#3a7d44','#d08b3e','#80475e']); ax.set_title('Comparação de desempenho'); ax.set_ylabel('Resultado'); ax.set_xlabel(''); ax.legend(loc='lower right'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(OUT,'comparacao_modelos.png'),dpi=180); plt.close()

# Console summary
print('SHAPES',train.shape,test1.shape,test2.shape,all_df.shape)
print('MISSING',int(all_df.isna().sum().sum()))
print('CLASS',all_df.Occupancy.value_counts().sort_index().to_dict())
print(res.to_string(index=False))
for n in res['Algoritmo']:
 print('\n',n,'CM',cms[n])
 print(pd.DataFrame(reports[n]).T.loc[['0','1'],['precision','recall','f1-score','support']])

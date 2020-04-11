clc;
clear all;
close all;
data=80;
x1=2*rand(data/2,1);
y1=2*rand(data/2,1);
x2=4*rand(data/2,1);
y2=3*rand(data/2,1);

x=[x1;x2];
y=[y1;y2];

xmedia=mean(x)
ymedia=mean(y)

xnuevos=x-xmedia*ones(data,1);
ynuevos=y-ymedia*ones(data,1);

figure
subplot(4,1,1);
plot(xnuevos,ynuevos,'+');
title('Datos Originales con la resta de la media');

subplot(4,1,2)
plot(x,y,'*');
title('Datos Originales');
#matriz de covarianza
MatrizCovarianza=cov(xnuevos,ynuevos)
#calcilo de los vectores propios  y valores de la matriz de covarianza
[V,D]=eig(MatrizCovarianza)
D=diag(D)
EugivalorMax=V(:,find(D==max(D)))
#escogiendo los componenes y vector futuro(datos finales)
datosFinales=EugivalorMax'*[xnuevos,ynuevos]'
subplot(4,1,3);
stem(datosFinales,'o');
title('Primera salida de datos PCA');
#clasificacion de datos finales
subplot(4,1,4);
title('Datos finales')
hold on 
for i=1:size(datosFinales,2)
  if datosFinales(i)>=0
    plot(x(i),y(i),'o')
    plot(x(i),y(i),'r*')
  else
     plot(x(i),y(i),'o')
     plot(x(i),y(i),'g*')
  end
end


#
# SMO-type decomposition method
#
# Referencia: Fan, Chen & Lin - Working set selection
#             using second order information for
#             training support vector machines (2005)
#
# Omar Trejo
# Modelos matemáticos y numéricos
# Prof. José Luis Morales
# ITAM, 2015
#
# Inputs:
# - Q: matriz Q_ij = y_i y_j K_ij; K matriz de kernel  (R^mxn)
# - y: vector entero en {-1, 1} con clasificación      (Z^n)
# - m: número de observaciones en los datos            (Z)
#
# NOTA: Este código todavía no está funcional.
#
function [] = smo_main(Q, y, m)

    # Parámetros
    eps = 1e-3;
    tau = 1e-12;

    # Inicialización
    alfa = zero(n, 1);
    gradiente = -ones(n, 1);

    while true

        i, j = seleccionar_conjunto_activo();

        if i == -1
            break;
        end

        a = Q[i][i] + Q[j][j] - 2*y[i]y[j]*Q[i][j];

        if a <= 0
            a = tau;
        end

        b = -y[i]*gradiente[i] + y[j]*gradiente[j];

        # Actualizar alfa
        alfa_vieja_i = alfa[i];
        alfa_vieja_j = alfa[j];
        alfa[i] += y[i]*b/a;
        alfa[j] += y[j]*b/a;

        # Proyectar alfa de regreso a la región factible
        suma = y[i]*alfa_vieja_i + y[j]*alfa_vieja_j;

        if alfa[i] > C
            alfa[i] = C;
        elseif alfa[i] < 0
            alfa[i] = 0;
        end

        alfa[j] = y[j]*(suma - y[i]*alfa[i]);

        if alfa[j] > C
            alfa[j] = C;
        elseif alfa[j] < 0
            alfa[j] = 0;
        end

        alfa[i] = y[i]*(suma - y[j]*alfa[j]);

        # Actualizar gradiente
        cambio_en_alfa_i = alfa[i] - alfa_vieja_i;
        cambio_en_alfa_j = alfa[j] - alfa_vieja_j;
        for t in 1:m
            gradiente[t] += Q[t][i]*cambio_en_alfa_i + Q[t][j]*cambio_en_alfa_j;
        end
    end
end

# TODO: arreglar parámetros
function [(i, j)] = seleccionar_conjunto_activo()
    #
    # Para i
    #
    i = -1;
    gradiente_maximo = -Inf;
    gradiente_minimo =  Inf;
    for t in 1:m
        if (y[t] == 1 && alfa[t] < C) || (y[t] == -1 && alfa[t] > 0)
            if -y[t]*gradiente[t] >= gradiente_maximo
                i = t;
                gradiente_maximo = -y[t]*gradiente[t];
            end
        end
    end

    #
    # Para j
    #
    j = -1;
    obj_min = Inf;
    for t in 1:m
        if (y[t] == 1 && alfa[t] > 0) || (y[t] == -1 && alfa[t] < C)
            b = gradiente_maximo + y[t]*gradiente[t];
            if -y[t]*gradiente[t] <= gradiente_minimo
                gradiente_minimo = -y[t]*gradiente[t];
            end
            if b > 0
                a = Q[i][i] + Q[t][t] - 2*y[i]*y[t]*Q[i][t];
                if a <= 0
                    a = tau;
                end
                if -b*b/a <= obj_min
                    j = t;
                    obj_min = -b*b/a;
                end
            end
        end
    end
    if gradiente_maximo - gradiente_minimo < eps
        return((-1, -1))
    else
        return((i, j))
    end
end

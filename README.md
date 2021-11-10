# Projeto20
Otimização de rotas de veículos com CVXPY

Otimização com CVXPY. Esta é uma tarefa do curso Otimização Discreta- https://www.coursera.org/learn/discrete-optimization/ O arquivo handout.pdf possui a descrição completa do problema e do formato de dados de entrada

Otimização de rotas de veículos.

O problema fornece a localização x,y de n clientes e um armazém de distribuição com m veículos. Cada cliente possui uma demanda 'Di' a ser atendida, cada veículo no armazém possui uma capacidade máxima de atendimento 'Ci'. O objetivo é otimizar a seleção de veículos de forma a atender todos os clientes existentes, minimizando
o somatório de distâncias percorrida por cada veículo. A capacidade máxima de atendimento de cada veículo não deve ser superada pelo somatório de demandas de clientes Di que estão em cada rota dos veículos.

Instruções:

Instale cvxpy: python -m pip install cvxpy

Instale o Gurobi: python -m pip install gurobipy

Instale o SCIP: https://github.com/scipopt/PySCIPOpt

Talvez voce precise de uma licença do Gurobi, no caso de problemas muito grandes

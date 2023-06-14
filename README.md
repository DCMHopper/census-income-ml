# census-income-ml

## Motive

To experiment with M1 acceleration in torch. To try training a basic torch model on a macbook, and see what kind of results I get. To flex a little ML skill while I'm not doing that work professionally. For fun, not profit.

## Execution

on Mac, run with command:
`PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py`

@ 200 epochs, 1e-7 learning rate, 0.9 momentum:
- Acc: 75.8%
- Prc: 44.0%
- Test loss: 0.271

@ 500 epochs, 1e-7 learning rate, 0.925 momentum:
- Acc: 80.7%
- Prc: 74.1%
- Test loss: 0.228

## Attribution/Sources

Data is from the "Adult/Census Income" dataset in the UCI Repository:

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
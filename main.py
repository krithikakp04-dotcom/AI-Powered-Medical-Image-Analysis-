"""AI Medical Image Analysis entry point."""

import textwrap


def main():
    message = textwrap.dedent(
        """
        AI Medical Image Analysis Project

        Use one of the project entry scripts directly:
          python train_model.py   # train the model
          python predict.py       # run prediction on the test image
          streamlit run app.py    # launch the web app

        Optional script:
          python evaluate_model.py
        """
    )
    print(message)


if __name__ == '__main__':
    main()

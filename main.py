""""
dash tool for pint project
"""

from app import app
from layout import create_layout  # Frontend logic
import callbacks                  # Importing ensures callbacks are registered

app.layout = create_layout()      # Assign layout to the app

if __name__ == "__main__":
    app.run_server(debug=True)





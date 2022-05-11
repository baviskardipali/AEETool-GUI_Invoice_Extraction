GUI_Invoice_Extraction
======================

This tool will assist in the automated and efficient extraction of template-free data from unstructured invoice documents. 

Pre-trained Neural Networks such as BERT, RoBERTa and DistilBERW are used to extract custom entities from invoice documents.

.. image:: https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg
     :alt: GUI Main Page
     
Installation
------------------

Poppler Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Poppler is a free software utility library for rendering Portable Document Format (PDF) documents. Among the list of very useful features, Poppler enables you to convert pdf files to images. Install poppler based on your operating system.

`Poppler installation guide <https://blog.alivate.com.au/poppler-windows/index.html>`__


Virtual environment setup(optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depending on the operating system in use, the user can setup a virtual environment to run this GUI.

To install the required libraries refer to requirements.txt file in the repository main page:

.. code:: shell

    pip install -r requirements.txt
    
Usage
-----

Steps
~~~~~~~~

1. Open a terminal and activate the virtual environment.
2. Run python app.py in the terminal in the GUI path to start the Flask interface.
3. Open a browser and enter the address as ``localhost:4000`` to enter the GUI interface.
4. Once the GUI is open, choose an invoice document and model respectively to upload the invoice to the website.
5. Click on the ``Create Model_Name`` button to get the output of the desired model.
6. On the next page, click on Export button to download the entities in an excel format.



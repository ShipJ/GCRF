Get raw data
============
Files are too large to permanently store in memory and so I suggest downloading them once, processing the data into
smaller, timestamped files providing easy access and ability to quickly insert/sample from the data. The time-stamped
files can again be stored once for the purposes of feature extraction, and then stored elsewhere (or deleted).

Processing Instructions
^^^^^^^^^^^^^^^^^^^^^^^
* Ivory Coast: download SET1.zip from the server (instructions in User Manual) and place into this directory. Proceed
to run process_raw_cdr.py, using 'civ' as the keyword argument.

* Senegal: similarly to above, grab the data set (SET1) from the server, but then manually expand it, and place the
expanded files in this directory. For some reason there are problems when automating the expansion procedure that I
cannot resolve so this is a solution for now. Use the keyword argument 'sen' to process the raw data. This data set
also included quantities of SMS messages in separate files, but these are not included in our research.





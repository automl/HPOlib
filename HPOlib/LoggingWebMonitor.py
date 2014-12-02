"""
LoggingWebMonitor - a central logging server and monitor.

Listens for log records sent from other processes running
in the same box or network.  Collects and saves them
concurrently in a log file.  Shows a summary web page with
the latest N records received.

Usage:

- Add a SocketHandler to your application::

    from logging.handlers import SocketHandler, DEFAULT_TCP_LOGGING_PORT
    socketh = SocketHandler(servername, DEFAULT_TCP_LOGGING_PORT)
    logging.getLogger('').addHandler(socketh)

  where servername is the host name of the logging server ('localhost'
  if run on the same box)

- Start an instance of this script (the logging server).
  This will open two listening sockets:

    - one at DEFAULT_TCP_LOGGING_PORT (9020), listening for
      logging events from your application

    - a web server at DEFAULT_TCP_LOGGING_PORT+1 (9021),
      showing a summary web page with the latest 200
      log records received.  That web page will be
      opened by default, using your preferred web browser.

- You may add additional handlers or filters to this script;
  see fileHandler below.

- Note that several separate processes *cannot* write to the same
  logging file; this script avoids that problem, providing
  the necesary isolation level.

- If you customize the logging system here, make sure `mostrecent`
  (instance of MostRecentHandler) remains attached to the root logger.
  
Author: Gabriel A. Genellina, based on code from Vinay Sajip and
doug.farrell; drastically cut down by Matthias Feurer

This original code is released here:
http://code.activestate.com/recipes/577025-loggingwebmonitor-a-central-logging-server-and-mon/
under the MIT (X11) license.
"""

import cPickle
from collections import deque
import logging
import logging.handlers
import SocketServer
import struct
import sys
import time
import threading


class MostRecentHandler(logging.Handler):
    'A Handler which keeps the most recent logging records in memory.'

    def __init__(self, max_records=200):
        logging.Handler.__init__(self)
        self.logrecordstotal = 0
        self.max_records = max_records
        try:
            self.db = deque([], max_records)
        except TypeError:
            # pre 2.6
            self.db = deque([])

    def emit(self, record):
        self.logrecordstotal += 1
        try:
            self.db.append(record)
            # pre 2.6
            while len(self.db) > self.max_records:
                self.db.popleft()
        except Exception:
            self.handleError(record)


# taken from the logging package documentation by Vinay Sajip

class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    'Handler for a streaming logging request'

    def handle(self):
        '''
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format.
        '''
        while 1:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return cPickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)
        for handler in logger.handlers:
            handler.flush()


class LoggingReceiver(SocketServer.ThreadingTCPServer):
    'Simple TCP socket-based logging receiver'

    logname = None

    def __init__(self, host='localhost',
                 port=None,
                 handler=LogRecordStreamHandler):
        if port is None:
            port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)


def main():
    mostrecent = MostRecentHandler()
    rootLogger = logging.getLogger('')
    rootLogger.setLevel(logging.DEBUG)
    rootLogger.addHandler(mostrecent)

    # # You may add additional handlers like this FileHandler
    ## that logs every message to a file
    ## named after this module name, with extension .log
    #
    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    #fileHandler = logging.FileHandler(os.path.splitext(__file__)[0] + '.log')
    #fileHandler.setFormatter(formatter)
    #rootLogger.addHandler(fileHandler)

    recv = LoggingReceiver()
    thr_recv = threading.Thread(target=recv.serve_forever)
    thr_recv.daemon = True
    print '%s started at %s' % (recv.__class__.__name__, recv.server_address)
    thr_recv.start()

    while True:
        try:
            time.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            recv.shutdown()
            break

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/python2
# coding: utf-8

from __future__ import print_function

import sys
import os
import urllib
import hmac
import json
import shutil
import tempfile

defaultserverurl = 'https://ws.voxygen.fr/ws'
ARGS_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'wsclient.args'))
CRED_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'wsclient.cred'))

class WSError(Exception):
        def __init__(self, url, code=None, msg=None):
                self.url = url
                self.code = code
                self.msg = msg
        def __str__(self):
                ret = self.url
                if self.code:
                        ret += " -> " + str(self.code)
                if self.msg:
                        ret += " ; " + str(self.msg)
                return ret

class WS:
        def __init__(self, url=None, user=None, password=None, debug=False):
                if user is None:
                        try:
                                with open(CRED_FILENAME, 'r') as f:
                                        user = f.readline().strip()
                                        if password is None:
                                                password = f.readline().strip() or None
                        except:
                                pass
                if url:
                        if not url.startswith('http://') and not url.startswith('https://'):
                                url = 'http://' + url
                        self.serverurl = url
                else:
                        self.serverurl = defaultserverurl
                self.user = user
                self.password = password
                self.debug = debug

        def wsopen(self, url, post, **params):
                noparam = params.pop('noparam',False)
                if noparam:
                        params = {}
                else:
                        if self.user is not None:
                                params['user'] = self.user
                        if self.password is not None:
                                params.pop('hmac', None)
                                HMAC=hmac.new(self.password)
                                for k,v in sorted(params.items()):
                                        HMAC.update("%s=%s" % (k,v))
                                params.update({'hmac':HMAC.hexdigest()})
                query = urllib.urlencode(params)
                if post:
                        body = query
                elif query:
                        url = "{}?{}".format(url, query)

                if self.debug:
                        if post:
                                print("POST:\n{}\n{!r}\n".format(url, body), file=sys.stderr)
                        else:
                                print("GET:\n{}\n".format(url), file=sys.stderr)

                class URLopener(urllib.FancyURLopener):
                        def http_error_default(self, url, fp, errcode, errmsg, headers):
                                return urllib.addinfourl(fp, headers, "http:" + url, errcode)
                try:
                        urllib._urlopener = URLopener()
                        if post:
                                resp = urllib.urlopen(url, body)
                        else:
                                resp = urllib.urlopen(url)
                except IOError as e:
                        raise WSError(url, msg=e)
                if self.debug:
                        print("RESPONSE:\n{}\n{}".format(resp.getcode(), resp.info()), file=sys.stderr)
                if resp.getcode() != 200:
                        raise WSError(url, resp.getcode(), resp.read())
                return resp


        def getjson(self, resp):
                ctype = resp.info().get('content-type')
                if ctype != 'application/json':
                        raise WSError(resp.geturl(), "'application/json' response expected, got {!r}".format(ctype))
                clength = resp.info().get('content-length', None)
                if clength is not None:
                        clength = int(clength)
                if clength is not None:
                        jstruct = resp.read(clength)
                else:
                        jstruct = resp.read()
                if self.debug:
                        print("JSON:\n{!r}\n".format(jstruct), file=sys.stderr)
                return json.loads(jstruct)

        audiotypes = ('application/octet-stream', 'audio/x-wav', 'audio/au', 'audio/mpeg', 'audio/ogg')
        def getaudiofile(self, resp, out):
                ctype = resp.info().get('content-type', 'application/octet-stream')
                if ctype not in self.audiotypes:
                        raise WSError(resp.geturl(), "unexpected type {!r} while expecting one of {!r}".format(ctype, self.audiotypes))
                clength = resp.info().get('content-length', None)
                if clength is not None:
                        clength = int(clength)
                if not out:
                        fd,out = tempfile.mkstemp(prefix='ws_', suffix='.audio')
                        os.close(fd)
                if isinstance(out, str):
                        outfile = open(out, 'wb')
                else:
                        outfile = out


                # copy to file until content-length is reached
                buflen=64*1024
                length = 0
                while 1:
                        buf = resp.read(buflen)
                        if not buf or length == clength:
                                break
                        if clength is not None:
                                if length + len(buf) > clength:
                                        buf = buf[:clength-length]
                        outfile.write(buf)
                        length += len(buf)

                return outfile, ctype, length


        def info(self, post=True):
                url = os.path.join(self.serverurl, 'info')
                resp = self.wsopen(url, post=post)
                struct = self.getjson(resp)
                resp.close()
                return struct

        def tts1(self, text, out=None, post=True, **params):
                url = os.path.join(self.serverurl, 'tts1')
                resp = self.wsopen(url, text=text, post=post, **params)
                outfile, ctype, length = self.getaudiofile(resp, out)
                resp.close()
                return outfile, ctype, length

        def tts2(self, text, out=None, post=True, **params):
                url = os.path.join(self.serverurl, 'tts2')
                resp = self.wsopen(url, text=text, post=post, **params)
                struct = self.getjson(resp)
                resp = self.wsopen(struct['url'], post=False, noparam=True)
                outfile, ctype, length = self.getaudiofile(resp, out)
                resp.close()
                return struct, outfile, ctype, length



if __name__ == '__main__':
        import argparse
        import pprint

	class ParamAction(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			params = getattr(namespace, 'params') or {}
			for param in values:
				if not '=' in param:
					parser.error("missing '=' in parameter %r" % param)
				key,value = param.split('=',1)
				params[key]=value
			setattr(namespace, self.dest, params)

        parser = argparse.ArgumentParser(description="Client to the Voxygen HTTP server.", epilog="Credentials may come from the command line under the form user=<user> and password=<password> Credentials may also come from a file called {} with the first two lines as user and password respectively Arguments may come from a file called {}".format(CRED_FILENAME, ARGS_FILENAME))
        parser.add_argument('-d', '--debug', action='store_true', help="debug webservice protocol")
        parser.add_argument('-q', '--quiet', action='store_true', help="quiet mode")

        parser.add_argument('-u', '--url', action='store', help="URL of the webservice (defaults to {})".format(defaultserverurl))
        parser.add_argument('params', nargs='*', action=ParamAction, metavar='param=value', help="request parameters. 'password' parameter if present is converted to hmac value. 'text' parameter may come from -i option")

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--POST', dest='method', action='store_const', const='POST', help="use POST method (default)", default='POST')
        group.add_argument('--GET', dest='method', action='store_const', const='GET', help="use GET method")

        group = parser.add_argument_group('input text / output signal')
        group.add_argument('-i', dest='inputfile', metavar='TEXTFILE', type=argparse.FileType('rb'), help="from TEXTFILE or stdin (-). Overwrites 'text=...' positional argument")
        group.add_argument('-o', dest='audiofile', metavar='AUDIOFILE', type=argparse.FileType('wb'), help="to AUDIOFILE or stdin (-)")
        if sys.platform.startswith('linux'):
                group.add_argument('--ecoute', action='store_true',  help="output to to speaker. incompatible with -o option")

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--info', dest='verb', action='store_const', const='info', help='get info from webservice account')
        group.add_argument('--tts1', dest='verb', action='store_const', const='tts1', help='get audio file in one pass (default)')
        group.add_argument('--tts2', dest='verb', action='store_const', const='tts2', help='get audio file in two pass')

        try:
                args,remaining = parser.parse_known_args(open(ARGS_FILENAME, 'r').read().split())
        except:
                args=None
                remaining=[]
        args = parser.parse_args(args=sys.argv[1:] + remaining, namespace=args)
 

        text = args.params.pop('text','')
        if args.inputfile:
                text = args.inputfile.read()

        play = None
        if args.ecoute:
                if args.audiofile:
                        parser.error("argument -o: not allowed with argument --ecoute")
                else:
                        import subprocess
                        header = args.params.get('header')
                        if header and header != 'wav-header' and not args.quiet:
                                print("changing parameter 'header' to 'wav-header'", file=sys.stderr)
                        args.params['header'] = 'wav-header'
                        coding = args.params.get('coding')
                        if coding and coding != 'lin' and not args.quiet:
                                print("changing parameter 'coding' to 'lin'", file=sys.stderr)
                        args.params['coding'] = 'lin'
                        try:
                                stderr = open(os.devnull, 'w') if args.quiet else None
                                play = subprocess.Popen(('play', '-t', 'wav', '-'), stdin=subprocess.PIPE, stderr=stderr)
                        except OSError:
                                print("Unable to find « play » command. Please install « sox » (universe)", file=sys.stderr)
                                sys.exit(1)
                        out = play.stdin
        else:
                out = args.audiofile

        user = args.params.pop('user', None)
        password = args.params.pop('password', None)
        try:
                ws = WS(args.url, user, password, args.debug and not args.quiet)
                if args.verb == 'info':
                        info = ws.info(post=args.method=='POST')
                        if not args.quiet:
                                pprint.pprint(info, stream=sys.stderr)
                elif args.verb == 'tts2':
                        struct,audiofile,ctype,length = ws.tts2(text=text, out=out, post=args.method=='POST', **args.params)
                        if not args.quiet:
                                pprint.pprint(struct, stream=sys.stderr)
                                print("\r{} bytes written in {} ({})         ".format(length,audiofile.name,ctype), file=sys.stderr)
                else:
                        audiofile,ctype,length = ws.tts1(text=text, out=out, post=args.method=='POST', **args.params)
                        if not args.quiet:
                                print("`\r{} bytes written in {} ({})        ".format(length,audiofile.name,ctype), file=sys.stderr)
                if play:
                        play.wait()
        except Exception, e:
                print(e, file=sys.stderr)
                if play:
                        play.kill()
                sys.exit(1)

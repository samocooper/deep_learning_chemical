
import * as socket from 'socket.io';
import { createWriteStream } from 'fs';
import { spawn } from 'child_process';
import { Readable as ReadableStream } from 'stream';
const ss = require('socket.io-stream');

const io = socket.listen(3000, () => {
  console.log('Server running');
});

io.on('connection', (client) => {
  console.log(`client ${client.id} connected`);
  const socketStream = ss(client);
  socketStream.on('file', (stream: ReadableStream) => {
    console.log('Uploading file');

    const fileName = `file_${client.id}`;
    const fileStream = createWriteStream(fileName);
    stream.pipe(fileStream);
    fileStream.on('close', () => {
      client.emit('file_uploaded', {name: fileName});
      train(client, fileName);
    });
  });

  client.emit('id', {id: client.id});

  client.on('event', (data: string) => {
    /* event */
  });

  client.on('disconnect', () => {
    /* disconnect */
  });
});

function train(socket: any, fileName: string) {
  console.log(`training ${fileName}`);
  const training = spawn('THEANO_FLAGS="cuda.root=/usr/local/cuda,device=cuda,floatX=float32" python', ['../main.py']);
  training.stdout.pipe(process.stdout);
  training.stderr.pipe(process.stderr);
  training.stdout.on('data', (data) => {
    socket.emit('training', data.toString());
  });
  training.stderr.on('data', (data) => {
    socket.emit('training', data.toString());
  })
}


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
      train(fileStream, fileName);
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

function train(socketStream: any, fileName: string) {
  console.log(`training ${fileName}`);
  const training = spawn('python', ['../main.py']);
  training.stdout.pipe(process.stdout);
  training.stderr.pipe(process.stderr);
  socketStream.emit('training', training.stdout, training.stderr);
}

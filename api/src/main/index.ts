
import * as socket from 'socket.io';
import { createWriteStream } from 'fs';
import { exec } from 'child_process';
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
  const process = exec('python ../main.py', (error, stdout, stderr) => {
    socketStream.emit('training', stdout, stderr);
  });
}

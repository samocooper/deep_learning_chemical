import hello from '../main/hello';
import * as test from 'tape';

test('it should greet the string provided', t => {
  t.plan(1);
  t.equals(hello('World'), 'Hello World');
});

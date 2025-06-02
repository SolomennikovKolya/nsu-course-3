definition Dogs;

class dog
	breed:               string;
	sizemin,sizemax:     integer(15..95);
	weightmin,weightmax: integer(1..100); 
	description:         set of string;

end;

end.
#ifndef NODE_YAML_H
#define NODE_YAML_H
enum status {
    SUCCESS = 1,
    FAILURE = 0
};

/* Our example parser states. */
const std::string ACTIONS{"actions"};
enum parse_state {
    STATE_START,    /* start state */
    STATE_STREAM,   /* start/end stream */
    STATE_DOCUMENT, /* start/end document */
    STATE_SECTION,  /* top level */

    STATE_ACTIONLIST,    /* action list */
    STATE_ACTIONVALUES,  /* action key-value pairs */
    STATE_ACTIONKEY,     /* action key */
    STATE_ACTIONNAME,    /* action name value */
    STATE_ACTION_VALUE,  /* all values encompassed as one*/
    STATE_STOP      /* end state */
};

//alias definitions
using ActionMap = std::map<std::string, std::string> ;
using ActionList = std::vector<ActionMap> ;
using Actions = std::map<std::string, ActionList> ;

/* Our application parser state data. */
struct parser_state {
    parse_state state;      /* The current parse state */
    ActionMap f;            /*  data elements. */
    ActionList actionlist;  /* List of action objects. */
    std::string keyname;    /* to retain key name from previous state */
};

int consume_event(struct parser_state *&s, yaml_event_t *event);
int parse_config(std::string filename);
#endif
